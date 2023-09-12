# ConBoTNet
This repository contains the codes for the paper: Supervised contrastive deep learning with bottleneck transformer improves MHC-II peptide binding affinity prediction
![avatar](./figure/Figure-1.png)
## Requirements
* python==3.9.16
* pytorch==1.12.1
* numpy==1.23.5
* scipy==1.10.0
* scikit-learn==1.2.1
* click==8.0.4
* ruamel.yaml==0.17.21
* logzero==1.7.0

## Experiments
```bash
# train and evaluation on independent test set.
python -u conbotnet_pretrain.py -d config/data.yaml --mode train
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode train

# 5 cross-validation
python -u conbotnet_pretrain.py -d config/data.yaml --mode 5cv
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode 5cv

# leave one molecule out cross-validation
python -u conbotnet_pretrain.py -d config/data.yaml --mode lomo
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode lomo

# binding core prediction (after model training)
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode binding

# seq2logo
python -u conbotnet_fine_tuning.py -d config/data.yaml --mode seq2logo
```

## Declaration
It is free for non-commercial use. For commercial use, please contact Long-Chen Shen (shenlc1995@njust.edu.cn).
