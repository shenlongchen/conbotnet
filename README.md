# ConBoTNet
This repository contains the codes for the paper: [**Supervised contrastive learning enhances MHC-II peptide binding affinity prediction**](https://www.sciencedirect.com/science/article/pii/S0957417425000855)
## Introduction 
Accurate prediction of major histocompatibility complex (MHC)-peptide binding affinity could provide essential insights into cellular immune responses and guide the discovery of neoantigens and personalized immunotherapies. Nevertheless, the existing deep learning-based approaches for predicting MHC-II peptide interactions fall short of satisfactory performance and offer restricted model interpretability. In this study, we propose a novel deep neural network, termed ConBoTNet, to address the above issues by introducing the designed supervised contrastive learning and bottleneck transformer extractors. Specifically, the supervised contrastive learning pre-training enhances the model’s representative and generalizable capabilities on MHC-II peptides by pulling positive pairs closer and pushing negative pairs further in the feature space, while the bottleneck transformer module focuses on MHC-II peptide interactions to precisely identify binding cores and anchor positions in an unsupervised manner. Extensive experiments on benchmark datasets under 5-fold cross-validation, leave-one-molecule-out validation, independent testing, and binding core prediction settings highlighted the superiority of our proposed ConBoTNet over current state-of-the-art methods. Data distribution analysis in the latent feature space demonstrated that supervised contrastive learning can aggregate MHC-II-peptide samples with similar affinity labels and learn common features of similar affinity. Additionally, we interpreted the trained neural network by associating the attention weights with peptides and innovatively find both well-established and potential peptide motifs. This work not only introduces an innovative tool for accurately predicting MHC-II peptide affinity, but also provides new insights into a new paradigm for modeling essential biological interactions, advancing data-driven discovery in biomedicine.
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
```tex
@article{shen2025supervised,
  title={Supervised contrastive learning enhances MHC-II peptide binding affinity prediction},
  author={Shen, Long-Chen and Liu, Yan and Liu, Zi and Zhang, Yumeng and Wang, Zhikang and Guo, Yuming and Rossjohn, Jamie and Song, Jiangning and Yu, Dong-Jun},
  journal={Expert Systems with Applications},
  pages={126463},
  year={2025},
  publisher={Elsevier}
}
```

## Declaration
It is free for non-commercial use. For commercial use, please contact Long-Chen Shen (shenlc1995@njust.edu.cn).
