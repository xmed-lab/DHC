import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default="fully")
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()

from utils import read_list, read_nifti
import torch
import torch.nn.functional as F
from utils.config import Config
config = Config(args.task)

if __name__ == '__main__':


    ids_list = read_list('test', task=args.task)
    results_all_folds = []
    txt_path = "./logs/"+args.exp+"/evaluation_res.txt"
    print("\n Evaluating...")
    fw = open(txt_path, 'w')
    for fold in range(1, args.folds+1):

        test_cls = [i for i in range(1, config.num_cls)]
        values = np.zeros((len(ids_list), len(test_cls), 2)) # dice and asd

        for idx, data_id in enumerate(tqdm(ids_list)):
            pred = read_nifti(os.path.join("./logs",args.exp, "fold"+str(fold), "predictions_"+str(args.cps),f'{data_id}.nii.gz'))
            if args.task == "amos":
                label = read_nifti(os.path.join(config.base_dir, 'labelsVa', f'{data_id}.nii.gz'))
            else:
                label = read_nifti(os.path.join(config.base_dir, 'labelsTr', f'label{data_id}.nii.gz'))
            label = label.astype(np.int8)
            dd, ww, hh = label.shape
            label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)


            if args.task == "amos":
                resize_shape = (config.patch_size[0] + config.patch_size[0] // 8,
                                config.patch_size[1] + config.patch_size[1] // 8,
                                config.patch_size[2] + config.patch_size[2] // 8)
                label = F.interpolate(label, size=resize_shape,mode='nearest')
            else:
                label = F.interpolate(label, size=(dd, ww//2, hh//2),mode='nearest')
            label = label.squeeze().numpy()

            for i in test_cls:
                pred_i = (pred == i)
                label_i = (label == i)
                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred == i, label == i) * 100
                    hd95 = metric.binary.asd(pred == i, label == i)
                    values[idx][i-1] = np.array([dice, hd95])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    dice, hd95 = 0, 128
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    dice, hd95 =  0, 128
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, hd95 =  1, 0

                values[idx][i-1] = np.array([dice, hd95])

        values_mean_cases = np.mean(values, axis=0)
        results_all_folds.append(values)
        fw.write("Fold" + str(fold) + '\n')
        fw.write("------ Dice ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,0],1)) + '\n')
        fw.write("------ ASD ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,1],1)) + '\n')
        fw.write('Average Dice:'+str(np.mean(values_mean_cases, axis=0)[0]) + '\n')
        fw.write('Average  ASD:'+str(np.mean(values_mean_cases, axis=0)[1]) + '\n')
        fw.write("=================================")
        print("Fold", fold)
        print("------ Dice ------")
        print(np.round(values_mean_cases[:,0],1))
        print("------ ASD ------")
        print(np.round(values_mean_cases[:,1],1))
        print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

    results_all_folds = np.array(results_all_folds)

    fw.write('\n\n\n')
    fw.write('All folds' + '\n')

    results_folds_mean = results_all_folds.mean(0)

    for i in range(results_folds_mean.shape[0]):
        fw.write("="*5 + " Case-" + str(ids_list[i]) + '\n')
        fw.write('\tDice:'+str(np.round(results_folds_mean[i][:,0],2).tolist()) + '\n')
        fw.write('\t ASD:'+str(np.round(results_folds_mean[i][:,1],2).tolist()) + '\n')
        fw.write('\t'+'Average Dice:'+str(np.mean(results_folds_mean[i], axis=0)[0]) + '\n')
        fw.write('\t'+'Average  ASD:'+str(np.mean(results_folds_mean[i], axis=0)[1]) + '\n')

    fw.write("=================================\n")
    fw.write('Final Dice of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]) + '\n')
    fw.write('Final ASD of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]) + '\n')
    print("=================================")
    print('Final Dice of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]))
    print('Final ASD of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]))
    std_dice = np.std(results_all_folds.mean(1).mean(1)[:,0])
    std_hd = np.std(results_all_folds.mean(1).mean(1)[:,1])

    fw.write('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 2)) +'±' +  str(round(std_dice,2)) + '\n')
    fw.write('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[1], 2)) +'±' +  str(round(std_hd,2)) + '\n')

    print('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 2)) +'±' +  str(round(std_dice,2)))
    print('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[1], 2)) +'±' +  str(round(std_hd,2)))



