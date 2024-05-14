import os
import sys
import logging
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-s', '--split', type=str, default='train')
parser.add_argument('--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
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
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import DC_and_CE_loss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_UD, RandomFlip_LR
from data.data_loaders import Synapse_AMOS
from utils.config import Config
import torch.nn as nn

config = Config(args.task)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

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

    # model
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()

    # model = kaiming_normal_init_weight(model)

    # dataloader
    db_train = Synapse_AMOS(task=args.task,
                            split=args.split,
                            num_cls=config.num_cls,
                            transform=transforms.Compose([
                        RandomCrop(config.patch_size, args.task),
                        RandomFlip_LR(),
                        RandomFlip_UD(),
                        ToTensor(),
                       ]))
    db_eval = Synapse_AMOS(task=args.task,
                           split=args.split_eval,
                           is_val=True,
                           num_cls=config.num_cls,
                           transform = transforms.Compose([
                                      CenterCrop(config.patch_size, args.task),
                                      ToTensor()
                                    ]))
    
    train_loader = DataLoader(
        db_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        worker_init_fn=seed_worker
    )
    eval_loader = DataLoader(db_eval, pin_memory=True)
    logging.info(f'{len(train_loader)} itertations per epoch')

    # optimizer, scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.base_lr, 
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )



    # loss function
    loss_func = DC_and_CE_loss()

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []

        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            image, label = fetch_data(batch)

            if args.mixed_precision:
                with autocast():
                    output = model(image)
                    del image
                    loss = loss_func(output, label)

                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                output = model(image)
                del image
                loss = loss_func(output, label)

                loss.backward()
                optimizer.step()
                # raise NotImplementedError

            loss_list.append(loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss', np.mean(loss_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')

        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        # print("lr:", np.round(optimizer.param_groups[0]['lr'], decimals=6))

        # lr_scheduler.step()
        # print("%.3e" %lr_scheduler.get_last_lr()[0])

        if epoch_num % 10 == 0:


            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in eval_loader:
                image, gt = fetch_data(batch)
                output = model(image)
                
                shp = output.shape
                gt = gt.long()
                y_onehot = torch.zeros(shp).cuda()
                y_onehot.scatter_(1, gt, 1)

                x_onehot = torch.zeros(shp).cuda()
                output = torch.argmax(output, dim=1, keepdim=True).long()


                # label_save = output[0][0].cpu().numpy().astype(np.int32)
                # label_save = sitk.GetImageFromArray(label_save)
                # sitk.WriteImage(label_save, '/home/xmli/hnwang/CLD_Semi/vis_test/label.nii.gz')


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
                torch.save(model.state_dict(), save_path)
                logging.info(f'save model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

    
    writer.close()


