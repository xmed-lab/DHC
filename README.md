## DHC (Dual-debiased Heterogeneous Co-training Framework)



### 1. Environment

This code has been tested with Python 3.6, PyTorch 1.8, torchvision 0.9.0, and CUDA 11.1 on Ubuntu 20.04.

### 2. Data Preparation

#### 2.1 Synapse
The MR imaging scans are available at https://www.synapse.org/#!Synapse:syn3193805/wiki/.
Please sign up and download the dataset. 

Put the data in anywhere you want then change the file paths in `config.py`.

Run `./code/data/preprocess.py` to 
- convert `.nii.gz` files into `.npy` for faster loading. 
- generate the train/validation/test splits
- generate the labeled/unlabeled splits 

Or use our pre-split files in `./synapse/splits/*.txt`. 

After preprocessing, the `./synapse_data/` folder should be organized as follows:

```shell
./synapse_data/
├── npy
│   ├── <id>_image.npy
│   ├── <id>_label.npy
├── splits
│   ├── labeled_20p.txt
│   ├── unlabeled_20p.txt
│   ├── train.txt
│   ├── eval.txt
│   ├── test.txt
│   ├── ...
```

#### 2.2 AMOS
The dataset can be downloaded from https://amos22.grand-challenge.org/Dataset/

Run `./code/data/preprocess_amos.py` to pre-process.

### 3. Training & Testing & Evaluating

Run the following commands for training, testing and evaluating.

```shell
bash train3times_seeds_20p.sh -c 0 -t synapse -m dhc -e '' -l 3e-2 -w 0.1
```
`20p` denotes training with 20% labeled data, you can change this to `2p`, `5p`, ... for 2%, 5%, ... labeled data.

Parameters:

`-c`: use which gpu to train

`-t`: task, can be `synapse` or `amos`

`-m`: method, `dhc` is our proposed method, other available methods including:
- cps
- uamt
- urpc
- ssnet
- dst
- depl
- adsh
- crest
- simis
- acisis
- cld

`-e`: name of current experiment

`-l`: learning rate

`-w`: weight of unsupervised loss



## License

This repository is released under MIT License.

