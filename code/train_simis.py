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
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
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
from models.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, EMA, print_func, xavier_normal_init_weight, kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
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
        weight_decay=3e-5,
        momentum=0.9,
        nesterov=True
    )
    return model, optimizer






class SimiS():
    def __init__(self, num_cls, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum

    def _cal_class_num(self, label_numpy):
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        return num_each_class.astype(np.float32)


    def _weight_wbg(self, num_each_class):
        P = (num_each_class+1e-8) / (np.sum(num_each_class)+1e-8)
        weight = (np.max(P) - P)
        return weight

    def init_weight(self, labeled_dataset):
        if labeled_dataset.unlabeled:
            raise ValueError
        num_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights = self._weight_wbg(num_each_class) * self.num_cls
        weights = torch.FloatTensor(weights).cuda()
        self.weights = torch.ones(self.num_cls).cuda() * self.num_cls
        self.weights = EMA(weights, self.weights, momentum=self.momentum)
        # self.weights = torch.FloatTensor(weights).cuda()
        return self.weights.data.cpu().numpy()

    def cal_cur_weight(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        num_each_class = self._cal_class_num(pseudo_label.data.cpu().numpy())

        weights = self._weight_wbg(num_each_class)

        weights =  torch.FloatTensor(weights).cuda()
        return weights * self.num_cls


    def get_weights(self, pseudo_label):
        cur_weights = self.cal_cur_weight(pseudo_label)
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        # self.weights = self.weights / torch.max(self.weights) * config.num_cls
        return self.weights










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
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)
    logging.info(optimizer_A)

    simis_A = SimiS(config.num_cls, do_bg=True, momentum=0.99)
    simis_B = SimiS(config.num_cls, do_bg=True, momentum=0.99)
    # make loss function
    weight_A = simis_A.init_weight(labeled_loader.dataset)
    weight_B = simis_B.init_weight(labeled_loader.dataset)
    # weight = torch.FloatTensor(weight).cuda()
    print("labeled data distribution:",print_func(weight_A))

    loss_func_A     = make_loss_function(args.sup_loss, weight=weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight=weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight=weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight=weight_B)

    # weight_A = weight_B = torch.FloatTensor(weight).cuda()

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

            if args.mixed_precision:
                with autocast():
                    output_A = model_A(image)
                    output_B = model_B(image)
                    del image

                    # sup (ce + dice)
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]


                    # cps (ce only)
                    max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                    max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()

                    weight_A = simis_A.get_weights(output_A)
                    weight_B = simis_B.get_weights(output_B)


                    loss_func_A.update_weight(weight_A)
                    loss_func_B.update_weight(weight_B)
                    cps_loss_func_A.update_weight(weight_A)
                    cps_loss_func_B.update_weight(weight_B)


                    loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)
                    loss_cps = cps_loss_func_A(output_A, max_B) + cps_loss_func_B(output_B, max_A)
                    # loss prop
                    loss = loss_sup + cps_w * loss_cps


                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()
                # if epoch_num>0:

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())

        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        # writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        writer.add_scalars('class_weights/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_A))), epoch_num)
        writer.add_scalars('class_weights/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_B))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')

        # logging.info(f'     cps_w: {cps_w}')
        # if epoch_num>0:
        logging.info(f"     Class Weights: {print_func(weight_A)}, lr: {round(get_lr(optimizer_A), 8)}")
        logging.info(f"     Class Weights: {print_func(weight_B)}")

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
            # '''
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

    writer.close()
