# G-Bert
Pre-training of Graph Augmented Transformers for Medication Recommendation

## Introduction
We have chosen “Pre-training of Graph Augmented Transformers for Medical Recommendation” paper, authored by Junyuan Shang, Tengfei Ma, Cao Xiao and Jimeng Sun, for our final project. The authors propose G-BERT graph learning model for medication recommendation. 

The advances in Deep Learning technologies, availability of resources and data have provided great opportunities in the field of medical recommendation. The authors argue that existing medication recommendation models often learn the resentations from the longitudinal EHR data from a small number of patients with multiple visits, ignoring patients with single visit (selection bias) and ignore the complex hierarchical relationships among diagnosis, medications, symptoms, and diseases (lack of hierarchical knowledge), which can lead to suboptimal recommendations. To address these issues and enhance the prediction and interpretability, they propose G-BERT graph learning model. The proposed model leverages medical knowledge encoded in large-scale medical corpora and single visit patient information to pre-train the GAT model to learn the underlying structure of medical concepts and relationships. The purpose of pre-training is mainly to leverage patient record with single visit and provide model trainings with good initializations.

The pre-trained GAT model is then fine-tuned on patient medical records with multiple visits, to personalize the medication recommendation for each patient. The model can capture not only the direct relationships between medications and symptoms but also the indirect relationships between medications through shared diseases or symptoms. As part of this course project work, we would like to reproduce the results which were published by the above mentioned authors in their original paper.

## Requirements
Below mentioned libraries are required to pre-train and fine-tune G-Bert.

- pytorch>=0.4
- python>=3.5
- torch_geometric==1.0.3
- numpy
- tqdm
- dill
- tensorboardX
- scikit-learn
- scipy
- torch_scatter==1.3.0
- torch_sparse==0.2.1
- torch_cluster

## Guide
We list the structure of this repo as follows:
```latex
.
├── code/
│   ├── bert_models.py % transformer models
│   ├── build_tree.py % build ontology
│   ├── config.py % hyperparameters for G-Bert
│   ├── graph_models.py % GAT models
│   ├── __init__.py
│   ├── predictive_models.py % G-Bert models
│   ├── run_alternative.sh % script to train G-Bert
│   ├── run_gbert.py % fine tune G-Bert
│   ├── run_gbert_side.py
│   ├── run_pretraining.py % pre-train G-Bert
│   ├── run_tsne.py # output % save embedding for tsne visualization
│   └── utils.py
├── data/
│   ├── data-multi-side.pkl 
│   ├── data-multi-visit.pkl % patients data with multi-visit
│   ├── data-single-visit.pkl % patients data with singe-visit
│   ├── dx-vocab-multi.txt % diagnosis codes vocabulary in multi-visit data
│   ├── dx-vocab.txt % diagnosis codes vocabulary in all data
│   ├── eval-id.txt % validation data ids
│   ├── px-vocab-multi.txt % procedure codes vocabulary in multi-visit data
│   ├── rx-vocab-multi.txt % medication codes vocabulary in multi-visit data
│   ├── rx-vocab.txt % medication codes vocabulary in all data
│   ├── test-id.txt % test data ids
│   └── train-id.txt % train data ids
└── saved/
    ├── GBert-predict/ % model files to reproduce our result
    │   ├── bert_config.json 
    │   └── pytorch_model.bin
    └── GBert-pretraining/ % model files to fine tune with multivisit data using pretrained model
        ├── bert_config.json 
        └── pytorch_model.bin
```
## Dataset Description
For pre-training and fine-tuning G-Bert, we are using the MIMIC-III synthetic data (pre-processed pickle files) made available in the git
repository [G-Bert Repo](https://github.com/jshang123/G-Bert). The statistics of the data is provided in the original paper [G-Bert](https://arxiv.org/pdf/1906.00346.pdf).

## Quick Test
To validate the performance of G-Bert, you can run the following script since we have provided the trained model binary file and well-preprocessed data.
```bash
cd code/

# G-Bert Trained Model to reproduce our results
python run_gbert.py --model_name GBert-predict-n --use_pretrain --pretrain_dir ../saved/GBert-predict --num_train_epochs 5 --do_train --do_test --graph

# G-Bert Model using pretrained model and fine tuning with multivisit data
python run_gbert.py --model_name GBert-predict-n --use_pretrain --pretrain_dir ../saved/GBert-pretraining --num_train_epochs 5 --do_train --do_test --graph
```
## Result Comparison
Below is the comparison of results between our reproduction study results and results published by the authors.

![image](https://user-images.githubusercontent.com/55331726/236712082-370b11bf-f858-407b-9b2f-652f289224c0.png)


## Citation 

Original paper:

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
- [G-Bert Repo](https://github.com/jshang123/G-Bert)

## Acknowledgement
Many thanks to the open source repositories and libraries to speed up our coding progress.

- [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)


