# G-Bert
Pre-training of Graph Augmented Transformers for Medication Recommendation

## Introduction
We have chosen “Pre-training of Graph Augmented Transformers for Medical Recommendation” paper, authored by Junyuan Shang, Tengfei Ma, Cao Xiao and Jimeng Sun, for our final project. The authors propose G-BERT graph learning model for medication recommendation. 

The advances in Deep Learning technologies, availability of resources and data have provided great opportunities in the field of medical recommendation. The authors argue that existing medication recommendation models often learn the resentations from the longitudinal EHR data from a small number of patients with multiple visits, ignoring patients with single visit (selection bias) and ignore the complex hierarchical relationships among diagnosis, medications, symptoms, and diseases (lack of hierarchical knowledge), which can lead to suboptimal recommendations. To address these issues and enhance the prediction and interpretability, they propose G-BERT graph learning model. The proposed model leverages medical knowledge encoded in large-scale medical corpora and single visit patient information to pre-train the GAT model to learn the underlying structure of medical concepts and relationships. The purpose of pre-training is mainly to leverage patient record with single visit and provide model trainings with good initializations.

The pre-trained GAT model is then fine-tuned on patient medical records with multiple visits, to personalize the medication recommendation for each patient. The model can capture not only the direct relationships between medications and symptoms but also the indirect relationships between medications through shared diseases or symptoms. As part of this course project work, we would like to reproduce the results which were published by the above mentioned authors in their original paper.

## Requirements
Below mentioned libraries are required to train and fine-tune G-Bert.

- pytorch>=0.4
- python>=3.5
- torch_geometric==1.0.3
- numpy
- tqdm
- dill
- tensorboardX
- sklearn
- scikit-learn
- scipy
- torch_scatter==1.3.0
- torch_sparse==0.2.1
- torch_cluster

## Guide
We list the structure of this repo as follows:
```latex
.
├── [4.0K]  code/
│   ├── [ 13K]  bert_models.py % transformer models
│   ├── [5.9K]  build_tree.py % build ontology
│   ├── [4.3K]  config.py % hyperparameters for G-Bert
│   ├── [ 11K]  graph_models.py % GAT models
│   ├── [   0]  __init__.py
│   ├── [9.8K]  predictive_models.py % G-Bert models
│   ├── [ 721]  run_alternative.sh % script to train G-Bert
│   ├── [ 19K]  run_gbert.py % fine tune G-Bert
│   ├── [ 19K]  run_gbert_side.py
│   ├── [ 18K]  run_pretraining.py % pre-train G-Bert
│   ├── [4.4K]  run_tsne.py # output % save embedding for tsne visualization
│   └── [4.7K]  utils.py
├── [4.0K]  data/
│   ├── [4.9M]  data-multi-side.pkl 
│   ├── [3.6M]  data-multi-visit.pkl % patients data with multi-visit
│   ├── [4.3M]  data-single-visit.pkl % patients data with singe-visit
│   ├── [ 11K]  dx-vocab-multi.txt % diagnosis codes vocabulary in multi-visit data
│   ├── [ 11K]  dx-vocab.txt % diagnosis codes vocabulary in all data
│   ├── [ 29K]  EDA.ipynb % jupyter version to preprocess data
│   ├── [ 18K]  EDA.py % python version to preprocess data
│   ├── [6.2K]  eval-id.txt % validation data ids
│   ├── [6.9K]  px-vocab-multi.txt % procedure codes vocabulary in multi-visit data
│   ├── [ 725]  rx-vocab-multi.txt % medication codes vocabulary in multi-visit data
│   ├── [2.6K]  rx-vocab.txt % medication codes vocabulary in all data
│   ├── [6.2K]  test-id.txt % test data ids
│   └── [ 23K]  train-id.txt % train data ids
└── [4.0K]  saved/
    └── [4.0K]  GBert-predict/ % model files to reproduce our result
        ├── [ 371]  bert_config.json 
        └── [ 12M]  pytorch_model.bin
```
### Preprocessing Data
We have released the preprocessing codes named data/EDA.ipynb to process data using raw files from MIMIC-III dataset. You can download data files from [MIMIC](https://mimic.physionet.org/gettingstarted/dbsetup/) and get necessary mapping files from [GAMENet](https://github.com/sjy1203/GAMENet).

### Quick Test
To validate the performance of G-Bert, you can run the following script since we have provided the trained model binary file and well-preprocessed data.
```bash
cd code/
# G-Bert Model using pretrained model and graph
python run_gbert.py --model_name GBert-predict-n --use_pretrain --pretrain_dir ../saved/GBert-pretraining --num_train_epochs 5 --do_train --do_test --graph
# G-Bert Model without using pretrained model and graph
python run_gbert.py --model_name GBert-predict-p-g-n --num_train_epochs 5 --do_train --do_test
# G-Bert Model without using pretrained model but using graph
python run_gbert.py --model_name GBert-predict-p-n --num_train_epochs 5 --do_train --do_test --graph
# G-Bert Model using pretrained model and not using graph
python run_gbert.py --model_name GBert-predict-g-n --use_pretrain --pretrain_dir ../saved/GBert-predict --num_train_epochs 5 --do_train --do_test
```
## Cite 

Original paper citiation:

```
@article{shang2019pre,
  title={Pre-training of Graph Augmented Transformers for Medication Recommendation},
  author={Shang, Junyuan and Ma, Tengfei and Xiao, Cao and Sun, Jimeng},
  journal={arXiv preprint arXiv:1906.00346},
  year={2019}
}
```

GATv2 paper:
```
@inproceedings{
  brody2022how,
  title={How Attentive are Graph Attention Networks? },
  author={Shaked Brody and Uri Alon and Eran Yahav},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=F72ximsx7C1}
}
```
Link to original paper repo:
- [G-Bert](https://github.com/jshang123/G-Bert)

## Acknowledgement
Many thanks to the open source repositories and libraries to speed up our coding progress.
- [GAMENet](https://github.com/sjy1203/GAMENet)
- [Bert_HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT)
- [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)


