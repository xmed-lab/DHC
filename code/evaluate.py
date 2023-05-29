import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm

from utils import read_list, read_nifti
from utils import config
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="fully")
    args = parser.parse_args()

    test_cls = [i for i in range(1, config.num_cls)]
    values = np.zeros((len(test_cls), 2)) # dice and asd
    ids_list = read_list('test')
    for data_id in tqdm(ids_list):
        pred = read_nifti(os.path.join("./logs",args.exp, "predictions",f'{data_id}.nii.gz'))
        label = read_nifti(os.path.join(config.base_dir, 'labelsTr', f'label{data_id}.nii.gz'))

        dd, ww, hh = label.shape
        label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label, size=(dd, ww//2, hh//2),mode='trilinear', align_corners=False)
        label = label.squeeze().numpy()

        for i in test_cls:

            pred_i = (pred == i)
            label_i = (label == i)
            if pred_i.sum() > 0 and label_i.sum() > 0:
                dice = metric.binary.dc(pred == i, label == i) * 100
                hd95 = metric.binary.hd95(pred == i, label == i)
                values[i - 1] += np.array([dice, hd95])

    values /= len(ids_list)
    print("====== Dice ======")
    print(np.round(values[:,0],1))
    print("====== HD ======")
    print(np.round(values[:,1],1))
    print(np.mean(values, axis=0)[0], np.mean(values, axis=0)[1])
