import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='cld')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('--crop_z', type=int, default=0)
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
import torch.nn.functional as F

from models.vnet import VNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, kaiming_normal_init_weight, xavier_normal_init_weight, print_func
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss, ClassDependent_WeightedCrossEntropyLoss, WeightedCrossEntropyLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_UD, RandomFlip_LR
from data.data_loaders import Synapse_AMOS
from utils.config import Config
config = Config(args.task)


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



def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size),
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
            worker_init_fn=seed_worker,
            drop_last=True
        ), len(dst)
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
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
    return model, optimizer



class ADSH:
    def __init__(self, num_cls=14, thresh_major=0.95, tot_iter=0):
        self.num_cls = num_cls
        # self.batch_size = batch_size
        self.thresh_major = thresh_major
        self.scores = torch.zeros(tot_iter, num_cls, config.patch_size[0], config.patch_size[1], config.patch_size[2]).float().cuda()
        self.cur_iter = 0
        self.tot_iter = tot_iter
        self.max_thresholds = thresh_major
        self.ema_thresholds = torch.zeros(num_cls).float().cuda()
        self.thresholds = torch.zeros(self.num_cls).float().cuda()
        # print("==",self.confs_all.size())

    def cal_weight(self, pseudo_label):
        onehot_shape = pseudo_label.shape
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        # print(torch.max(pseudo_label))
        weight = np.zeros(config.num_cls)
        for i in range(onehot_shape[0]):
            label = pseudo_label[i].data.cpu().numpy().reshape(-1)
            tmp, _ = np.histogram(label, range(config.num_cls + 1))
            weight += tmp
        weight =  torch.FloatTensor(weight).cuda()
        majority_cls = torch.argmax(weight, keepdim=True)
        return weight, majority_cls

    def get_thresholds(self, output):
        bsz = output.shape[0]
        if self.cur_iter < self.tot_iter-1:
            self.scores[self.cur_iter:self.cur_iter+bsz] = F.softmax(output.detach(), dim=1)
            self.cur_iter+=bsz
            return self.thresholds

        else:
            # print(self.scores.shape)
            b,c = self.scores.shape[0], self.scores.shape[1]
            scores = self.scores.view(b, c, -1)
            voxel_num, majority_cls = self.cal_weight(scores)
            score_major = scores[:,majority_cls].squeeze()
            scores_thresh = torch.where(score_major>self.thresh_major, torch.ones_like(score_major), torch.zeros_like(score_major))
            voxel_num_thresh = torch.sum(scores_thresh)
            rho = (voxel_num_thresh+1e-8) / (voxel_num[majority_cls]+1e-8)
            scores_list = []
            for t in range(scores.shape[0]):
                scores_list.append(scores[t])
            scores_stack = torch.cat(scores_list, dim=-1)

            for i in range(0, self.num_cls):
                sort_scores, _ = torch.sort(scores_stack[i], dim=0, descending=True)
                self.thresholds[i] = sort_scores[int(rho*voxel_num[i])]
                # if self.thresholds[i] > self.max_thresholds:
                #     self.thresholds[i] = self.max_thresholds
            self.cur_iter=0
            return self.thresholds

    def cal_sampling_mask(self, output, thresholds):
        confidence_map = F.softmax(output.detach(), dim=1)
        sample_map = torch.zeros_like(output).float()
        for idx in range(config.num_cls):
            cur_conf_map = (confidence_map[:,idx] > thresholds[idx]) * 1.0
            sample_map[:,idx] = cur_conf_map
        return sample_map



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
    logging.info(f'patch size: {config.patch_size}')

    # make data loader
    unlabeled_loader, u_len = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader, _ = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)

    # make loss function
    # weight,_ = labeled_loader.dataset.weight()
    loss_func = make_loss_function(args.sup_loss)
    if args.cps_loss == 'wce':
        cps_loss_func = ClassDependent_WeightedCrossEntropyLoss()
    else:
        raise ValueError

    # confidence bank
    # print(u_len)
    adsh_A = ADSH(num_cls=config.num_cls, thresh_major=0.9, tot_iter=u_len//4)
    adsh_B = ADSH(num_cls=config.num_cls, thresh_major=0.9, tot_iter=u_len//4) # <--

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            # print(image.shape)

            if args.mixed_precision:
                with autocast():
                    output_A = model_A(image)
                    output_B = model_B(image)
                    del image

                    # split labeled and unlabeled output
                    # output_A_l = output_A[:tmp_bs, ...]
                    # output_B_l = output_B[:tmp_bs, ...]
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]

                    # sup (ce + dice)
                    # <--
                    # update confidence bank

                    # <--
                    # loss function
                    loss_sup = loss_func(output_A_l, label_l) + loss_func(output_B_l, label_l)

                    # cps (ce only)
                    max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                    max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()

                    # update category sampling rate
                    thresholds_A = adsh_A.get_thresholds(output_A_u.detach())
                    thresholds_B = adsh_B.get_thresholds(output_B_u.detach())
                    # print(thresholds_A)

                    # print("\n",sam_rate_A)
                    # <--
                    # update sampling mask (without gt)
                    sample_map_A = adsh_A.cal_sampling_mask(output_A.detach(), thresholds_A)
                    sample_map_B = adsh_B.cal_sampling_mask(output_B.detach(), thresholds_B)
                    # print(sample_map_A.shape)

                    # loss function
                    loss_cps = cps_loss_func(output_A, max_B.detach(), sample_map_B) + cps_loss_func(output_B, max_A.detach(), sample_map_A)

                    # loss prop
                    loss = loss_sup + cps_w * loss_cps

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()
            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())

        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        writer.add_scalars('thresholds/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(thresholds_A))), epoch_num)
        writer.add_scalars('thresholds/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(thresholds_B))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')
        # logging.info(f'    - confidence {confs_A.get_sample_rate()}, {confs_B.get_confs()}')
        logging.info(f'    - thresholds A {print_func(thresholds_A)}')
        logging.info(f'    - thresholds B {print_func(thresholds_B)}')

        # lr_scheduler_A.step()
        # lr_scheduler_B.step()

        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:
            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    # output = model_A(image)
                    output = (model_A(image) + model_B(image)) / 2.0
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

            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break
            # '''

    writer.close()
