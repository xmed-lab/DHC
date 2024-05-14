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
parser.add_argument('-w', '--cps_w', type=float, default=1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=False) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import contextlib
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.vnet import VNet4SSNet
# from models.unet import unet_3D_ds
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
config = Config(args.task)
from torch.nn import functional as F



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


def make_model_all():
    model = VNet4SSNet(
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


def contrastive_class_to_class_learned_memory(model, features, class_labels, num_classes, memory):
    """
    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classesin the dataet
        memory: memory bank [List]
    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0
    for c in range(num_classes):
        # get features of an specific class
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        memory_c = memory[c] # N, 256

        # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
        selector = model.__getattr__('contrastive_class_selector_' + str(c))
        selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()

            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1) # N, 256
            features_c_norm = F.normalize(features_c, dim=1) # M, 256

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - similarities # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)


            # now weight every sample

            learned_weights_features = selector(features_c.detach()) # detach for trainability
            learned_weights_features_memory = selector_memory(memory_c)

            # self-atention in the memory featuers-axis and on the learning contrsative featuers-axis
            learned_weights_features = torch.sigmoid(learned_weights_features)
            rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)) * learned_weights_features
            rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
            distances = distances * rescaled_weights


            learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            rescaled_weights_memory = (learned_weights_features_memory.shape[0] / learned_weights_features_memory.sum(dim=0)) * learned_weights_features_memory
            rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_weights_memory


            loss = loss + distances.mean()

    return loss / num_classes


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d


class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = SoftDiceLoss(smooth=1e-8, do_bg=True)

    def forward(self, model, x):
        with torch.no_grad():
            # print(model(x).size())
            pred= F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device) ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d) ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds


class FeatureMemory:

    def __init__(self,  elements_per_class=32, n_classes=2):
        self.elements_per_class = elements_per_class
        self.memory = [None] * n_classes
        self.n_classes = n_classes

    def add_features_from_sample_learned(self, model, features, class_labels):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size
        Returns:
        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        elements_per_class = self.elements_per_class

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
            features_c = features[mask_c, :] # get features from class c
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        # sort them
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        # get features with highest rankings
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                else:
                    new_features = features_c.cpu().numpy()

                self.memory[c] = new_features


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
    prototype_memory = FeatureMemory(elements_per_class=32, n_classes=config.num_cls)
    # model_B, optimizer_B = make_model_all()
    # model_A = kaiming_normal_init_weight(model_A)
    # model_B = xavier_normal_init_weight(model_B)

    logging.info(optimizer)

    # make loss function
    # weight,_ = labeled_loader.dataset.weight()
    # print(weight)
    # print(sum(weight))

    dice_loss = SoftDiceLoss(smooth=1e-8, do_bg=True)
    adv_loss=VAT3d(epi=10.0)



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
                    outputs, embedding = model(image)
                    outputs_soft = F.softmax(outputs, dim=1)
                    labeled_features = embedding[:tmp_bs,...]
                    unlabeled_features = embedding[tmp_bs:,...]
                    y = outputs_soft[:tmp_bs]
                    true_labels = label_l.squeeze(1)

                    _, prediction_label = torch.max(y, dim=1)
                    _, pseudo_label = torch.max(outputs_soft[tmp_bs:], dim=1)  # Get pseudolabels

                    # print(true_labels.size())
                    # print(prediction_label.size())

                    mask_prediction_correctly = ((prediction_label == true_labels).float() * (prediction_label > 0).float()).bool()
                    # print(mask_prediction_correctly.size())
                    ### select the correct predictions and ignore the background class

                    # Apply the filter mask to the features and its labels
                    labeled_features = labeled_features.permute(0, 2, 3, 4, 1)
                    # print(labeled_features.size())
                    labels_correct = true_labels[mask_prediction_correctly]
                    labeled_features_correct = labeled_features[mask_prediction_correctly, ...]

                    # print(labeled_features_correct.size())

                    # get projected features
                    with torch.no_grad():
                        model.eval()
                        proj_labeled_features_correct = model.projection_head(labeled_features_correct)
                        model.train()

                    # updated memory bank
                    prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct, labels_correct)

                    labeled_features_all = labeled_features.reshape(-1, labeled_features.size()[-1])
                    labeled_labels = true_labels.reshape(-1)

                    # get predicted features
                    proj_labeled_features_all = model.projection_head(labeled_features_all)
                    pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)

                    # Apply contrastive learning loss
                    loss_contr_labeled = contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labeled_labels, config.num_cls, prototype_memory.memory)


                    unlabeled_features = unlabeled_features.permute(0, 2, 3, 4, 1).reshape(-1, labeled_features.size()[-1])
                    pseudo_label = pseudo_label.reshape(-1)

                    # get predicted features
                    proj_feat_unlabeled = model.projection_head(unlabeled_features)
                    pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

                    # Apply contrastive learning loss
                    loss_contr_unlabeled = contrastive_class_to_class_learned_memory(model, pred_feat_unlabeled, pseudo_label, config.num_cls, prototype_memory.memory)

                    loss_seg = F.cross_entropy(outputs[:tmp_bs], true_labels)

                    # print(y.size())

                    loss_seg_dice = dice_loss(y, true_labels)

                    loss_lds = adv_loss(model, image)

                    # iter_num = iter_num + 1


                    loss =  loss_seg_dice + cps_w * (loss_lds + 0.1 * (loss_contr_labeled + loss_contr_unlabeled))



                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            # loss_sup_list.append(supervised_loss.item())
            # loss_cps_list.append(consistency_loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        # writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        # writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}, cpsw:{cps_w} lr: {get_lr(optimizer)}')


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
