import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='cps')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='ce')
parser.add_argument('--sup_loss', type=str, default='wce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=False) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import math
from models.vnet import VNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, print_func,kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss, WeightedCrossEntropyLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
config = Config(args.task)
from torch.nn import functional as F


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w






def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size, args.task),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size, args.task),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all(ema=False):
    # model = unet_3D(n_classes=config.num_cls, in_channels=1).cuda()
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    if ema:
        for param in model.parameters():
            param.detach_()
    return model, optimizer





class ConfidenceArray:
    def __init__(self, num_cls=5, momentum=0.999, update_iter=1):
        self.num_cls = num_cls
        self.momentum = momentum
        self.tot_iter = update_iter
        self.cur_iter = 0
        self.entropy = torch.zeros(self.num_cls).float().cuda()
        self.confidence = torch.zeros(self.num_cls).float().cuda()
        self.variance = torch.zeros(self.num_cls).float().cuda()
        self.CC = torch.zeros(self.num_cls).float().cuda()
        self.cur_CC = torch.zeros(self.num_cls).float().cuda()
        self.top_m = 12
        self.P_CCF = 0
        self.P_FR = 0.632
        self.lambda_ = 2.5
        self.beta = 1.5
        self.sam_rate = torch.ones(self.num_cls).float().cuda()


    def record_data(self, output, gt):
        scores = F.softmax(output, dim=1)
        scores_max, _ = torch.max(scores, dim=1)
        for ind in range(0, self.num_cls):
            conf_map_sup = scores[:, ind, ...]
            voxels_cur_class = (gt==ind).sum() + 1e-12
            self.entropy[ind] += torch.sum(conf_map_sup.unsqueeze(1) * torch.log(scores+1e-12) * (gt==ind).unsqueeze(1)) / voxels_cur_class
            # print(self.entropy[ind])
            self.confidence[ind] += torch.sum(conf_map_sup * (gt==ind))  / voxels_cur_class
            # print(scores_max.shape, conf_map_sup.shape)
            self.variance[ind] += torch.sum((scores_max - conf_map_sup)  * (gt==ind)) / voxels_cur_class
            # print(self.entropy[ind], self.confidence[ind], self.variance[ind])
        self.cur_iter += 1



    # def gompertz(self, indicator):
    #
    #     indicator_norm = torch.zeros_like(indicator)
    #     # indicator_norm = torch.softmax(indicator, dim=1)
    #     # print(indicator_norm)
    #
    #     for i in range(0, indicator.shape[0]):
    #         indicator[i] = (indicator[i] - indicator[i].min()) / (indicator[i].max() - indicator[i].min())
    #         indicator_norm[i] = indicator[i] / indicator[i].sum()
    #     # print(indicator_norm)
    #     # indicator/=3
    #     return indicator_norm, 1 - torch.exp(-torch.exp(-2 * indicator_norm))


    def fuzzy_rank(self, CF, top):
        R_L = np.zeros(CF.shape)
        for i in range(CF.shape[0]):
            for k in range(CF.shape[1]):
                R_L[i][k] = 1 - math.exp(-math.exp(-2.0*CF[i][k]))  #Gompertz Function

        K_L = self.P_FR*np.ones(shape = R_L.shape) #initiate all values as penalty values
        for i in range(R_L.shape[0]):
            for k in range(top):
                a = R_L[i]
                idx = np.where(a==np.partition(a, k)[k])
                #if sample belongs to top 'k' classes, R_L =R_L, else R_L = penalty value
                K_L[i][idx] = R_L[i][idx]

        return K_L

    def CFS_func(self, CF, K_L):
        H = CF.shape[0] #no. of classifiers
        for f in range(CF.shape[0]):
            idx = np.where(K_L[f] == self.P_FR)
            CF[f][idx] = self.P_CCF
        CFS = 1 - np.sum(CF,axis=0) / H
        return CFS


    def cal_sampling_rate(self):

        if self.cur_iter == self.tot_iter:

            confs_array = torch.stack([self.entropy, self.confidence, self.variance], dim=0)
            # print("----")
            # print(self.entropy)
            R_norm = F.softmax(confs_array / self.cur_iter, dim=1).cpu().data.numpy()
            # print("")


            R = self.fuzzy_rank(R_norm, self.top_m)

            # R_norm, R = self.gompertz(confs_array)
            # R_c_norm, R_c = self.gompertz(self.confidence)
            # R_v_norm, R_v = self.gompertz(self.variance)

            # CCF = torch.stack([R_e_norm, R_c_norm, R_v_norm], dim=1)
            # FR = torch.stack([R_e, R_c, R_v], dim=1)
            # print(R_norm)
            # print(R)

            # print(np.argmax(R_norm, axis=1), np.argmin(R_norm, axis=1))
            # print(np.argmax(R, axis=1), np.argmin(R, axis=1))

            FR = R.sum(0)
            CCF = self.CFS_func(R_norm, R)



            cur_CC = torch.FloatTensor(CCF * FR).cuda()

            cur_CC = cur_CC / cur_CC.max()
            # self.cur_CC[0] = 1

            # self.pre_CC = self.CC
            # print(self.cur_CC)
            self.CC = EMA(cur_CC, self.CC, momentum=self.momentum)
            # print(self.CC)

            self.entropy = torch.zeros(self.num_cls).float().cuda()
            self.confidence = torch.zeros(self.num_cls).float().cuda()
            self.variance = torch.zeros(self.num_cls).float().cuda()
            self.cur_iter = 0
            # print(self.CC)

            sam_rate = 1 - self.CC
            # sam_rate = (sam_rate - sam_rate.min()) / (sam_rate.max() - sam_rate.min())
            # print(sam_rate)
            # self.sam_rate[0] = 0
            self.sam_rate = sam_rate / (sam_rate.max()+1e-12)
            # print(cur_sam_rate)
            # self.sam_rate = sam_rate_fg
            # print(sam_rate)
            # self.sam_rate = self.sam_rate / self.sam_rate.max()
            self.sam_rate = torch.pow(self.sam_rate, self.lambda_)
        return self.sam_rate

    def cal_sampling_mask(self,output, min_sr=0):
        pred_map = torch.argmax(output, dim=1).float()
        pred_map_max, _ = torch.max(torch.softmax(output, dim=1), dim=1)
        sample_map = torch.zeros_like(pred_map).float()
        vol_shape = pred_map.shape
        for idx in range(config.num_cls):
            prob = 1 - self.sam_rate[idx]
            if idx >= 1 and prob > (1 - min_sr):
                prob = (1 - min_sr)
            rand_map = torch.rand(vol_shape).cuda() * (pred_map == idx)
            rand_map = (rand_map > prob) * 1.0
            sample_map += rand_map
        sample_map = sample_map * torch.pow(pred_map_max, self.beta)
        # sample_map = sample_map/(sample_map.sum() + 1e-12)
        return sample_map

# def cal_sampling_rate(confs, gamma=2.5):
#     sam_rate = (1 - confs)
#     sam_rate = sam_rate / (torch.max(sam_rate)+1e-12)
#     sam_rate = sam_rate ** gamma
#     return sam_rate

# def cal_sampling_mask(output, sam_rate, min_sr=0.0):
#     pred_map = torch.argmax(output, dim=1).float()
#     pred_map_max, _ = torch.max(output, dim=1)
#     sample_map = torch.zeros_like(pred_map).float()
#     vol_shape = pred_map.shape
#     for idx in range(config.num_cls):
#         prob = 1 - sam_rate[idx]
#         if idx >= 1 and prob > (1 - min_sr):
#             prob = (1 - min_sr)
#         rand_map = torch.rand(vol_shape).cuda() * (pred_map == idx)
#         rand_map = (rand_map > prob) * 1.0
#         sample_map += rand_map
#     return sample_map


if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)


    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model, optimizer = make_model_all()
    ema_model, _ = make_model_all(ema=True)
    model = kaiming_normal_init_weight(model)
    ema_model = kaiming_normal_init_weight(ema_model)
    confs_arr = ConfidenceArray(num_cls=config.num_cls, momentum=0.99, update_iter=len(unlabeled_loader))


    logging.info(optimizer)


    loss_func = make_loss_function(args.sup_loss)
    # cps_loss_func = make_loss_function(args.cps_loss)
    # cps_loss_func = ClassDependent_WeightedCrossEntropyLoss()
    cps_loss_func = WeightedCrossEntropyLoss()


    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model.train()
        ema_model.train()
        # model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer.zero_grad()
            # optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    # noise = torch.clamp(torch.randn_like(
                    #     image_u) * 0.1, -0.2, 0.2)
                    # ema_inputs = image_u


                    outputs = model(image)
                    # outputs_soft = torch.softmax(outputs, dim=1)

                    output_l, output_u = outputs[:tmp_bs, ...], outputs[tmp_bs:, ...]

                    with torch.no_grad():
                        ema_output = ema_model(image_u)
                        pseudo_label = torch.argmax(ema_output.detach(), dim=1, keepdim=True).long()


                    loss_sup = loss_func(output_l, label_l)

                    confs_arr.record_data(output_l.detach(), label_l.detach())

                    sample_rate = confs_arr.cal_sampling_rate()

                    # print("CC",CC_score)

                    # print(sample_rate)

                    sample_map = confs_arr.cal_sampling_mask(output_u.detach())
                    # print(sample_map.shape)
                    # <--
                    # loss function

                    loss_cps = cps_loss_func(output_u, pseudo_label, sample_map)

                    loss = loss_sup + cps_w * loss_cps

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
                update_ema_variables(model, ema_model, 0.99, epoch_num)

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}, cpsw:{cps_w} lr: {get_lr(optimizer)}')
        logging.info(f'    - sampling rate {print_func(sample_rate)}')


        # lr_scheduler_A.step()
        # lr_scheduler_B.step()

        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    # output = model_A(image)
                    output = model(image)
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)

                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
