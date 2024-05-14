import torch
import numpy as np
import torch.nn.functional as F


class CenterCrop(object):
    def __init__(self, output_size, task):
        self.output_size = output_size
        self.task = task

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or \
                       image.shape[1] <= self.output_size[1] or \
                       image.shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}


        if w1 is None:
            (w, h, d) = image.shape
            w1 = int(round((w - self.output_size[0]) / 2.))
            if self.task == 'synapse':
                h1 = int(round((h//2 - self.output_size[1]) / 2.))
                d1 = int(round((d//2 - self.output_size[2]) / 2.))
            else:
                h1 = int(round((h - self.output_size[1]) / 2.))
                d1 = int(round((d - self.output_size[2]) / 2.))


        for key in sample.keys():
            item = sample[key]
            if self.task == 'synapse':
                dd, ww, hh = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':
                    item = F.interpolate(item, size=(dd, ww // 2, hh // 2), mode='trilinear', align_corners=False)
                    # print("img",item.shape)
                else:
                    item = F.interpolate(item, size=(dd, ww // 2, hh // 2), mode="nearest")
                    # print("lbl",item.shape)
                item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item

        return ret_dict

class RandomCrop(object):
    '''
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, output_size, task):
        self.output_size = output_size
        self.task = task

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]
        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
        
        w1, h1, d1 = None, None, None
        ret_dict = {}

        if w1 is None:
            (w, h, d) = image.shape
            w1 = np.random.randint(0, w - self.output_size[0])
            if self.task == 'synapse':
                h1 = np.random.randint(0, h //2 - self.output_size[1])
                d1 = np.random.randint(0, d //2 - self.output_size[2])
            else:
                h1 = np.random.randint(0, h - self.output_size[1])
                d1 = np.random.randint(0, d - self.output_size[2])

        for key in sample.keys():
            item = sample[key]

            # print(item.shape)
            if self.task == 'synapse':
                dd, ww, hh = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':
                    item = F.interpolate(item, size=(dd, ww//2, hh//2),mode='trilinear', align_corners=False)
                    # print("img",item.shape)
                else:
                    item = F.interpolate(item, size=(dd, ww//2, hh//2), mode="nearest")
                    # print("lbl",item.shape)
                item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            item = item[w1:w1+self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            # print(item.shape)
            ret_dict[key] = item
        
        return ret_dict


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = np.flip(img,1).copy()
        return img

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = np.flip(img, 2).copy()
        return img

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                # print(item.max())
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            elif key == 'label':
                # item[item>config.num_cls-1]=0
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError(key)
        # print(ret_dict['image'].shape)
        
        return ret_dict
