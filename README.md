# ConBoTNet
This repository contains the codes for the paper: **Supervised contrastive deep learning with bottleneck transformer improves MHC-II peptide binding affinity prediction**
## Introduction 
Accurate identification of major histocompatibility complex (MHC)-peptide binding affinity provides important insights into cellular immune responses and guides the discovery of neoantigens and immunotherapy. However, existing state-of-the-art deep learning methods for MHC-II peptide binding prediction do not achieve satisfactory performance and have limited model interpretability. Here, we propose ConBoTNet, a new deep learning-based approach by integrating the supervised contrastive learning and bottleneck transformer modules. The pre-training strategy of supervised contrastive learning endows the model with stable and reliable initial model weights. Meanwhile, the bottleneck transformer module can effectively identify the binding core and exact anchor positions from the interaction features between the MHC-II molecule and peptide. In addition, we analyzed the weights assigned by the model to peptide sequences to better interpret and understand the influence of binding cores on the binding affinity. Analysis of the pre-trained features also showed that the supervised contrastive learning can improve the model performance and accelerate the convergence. Benchmarking experiments on multiple large-scale data sets show that ConBoTNet significantly outperformed five state-of-the-art methods on 5-fold cross-validation, leave-one-molecule-out, independent test and binding core prediction. Taken together, the effectiveness of ConBoTNet is illustrated through multiple tasks including binding affinity prediction, binding core prediction, important anchor identification and model interpretability. This work also sheds light on how deep learning framework can be designed to model important biological interaction events such as MHC-II peptide binding and enable data-driven biomedical knowledge discovery.
![figure](./figure/Figure-1.png)
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

## Citation
Long-Chen Shen, Yan Liu, Zi Liu, Yumeng Zhang, Zhikang Wang, Yuming Guo, Jamie Rossjohn, Jiangning Song, and Dong-Jun Yu. *Supervised contrastive deep learning with bottleneck transformer improves MHC-II peptide binding affinity prediction*.

## Declaration
It is free for non-commercial use. For commercial use, please contact Long-Chen Shen (shenlc1995@njust.edu.cn).
