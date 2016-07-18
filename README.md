# Segmentation from Natural Language Expressions
This repository contains the Caffe reimplementation of the following paper:

* R. Hu, M. Rohrbach, T. Darrell, *Segmentation from Natural Language Expressions*. in arXiv:1603.06180, 2016. ([PDF](http://arxiv.org/pdf/1603.06180))
```
@article{hu2016segmentation,
  title={Segmentation from Natural Language Expressions},
  author={Hu, Ronghang and Rohrbach, Marcus and Darrell, Trevor},
  journal={arXiv preprint arXiv:1603.06180},
  year={2016}
}
```

Project Page: http://ronghanghu.com/text_objseg  

## Installation
1. Install Caffe following the instructions [here](http://caffe.berkeleyvision.org/installation.html).
2. Download this repository or clone with Git, and then `cd` into the root directory of the repository.

## Training and evaluation on ReferIt Dataset

### Download dataset and VGG network
1. Download ReferIt dataset:  
`./referit/referit-dataset/download_referit_dataset.sh`.
2. Download the caffemodel for [VGG-16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) network parameters trained on ImageNET 1000 classes.

### Training
3. You may need to add the repository root directory to Python's module path:  
```
export PYTHONPATH=/path/to/text_objseg_caffe/:$PYTHONPATH
```
4. Build training batches for bounding boxes:  
`python referit/build_training_batches_det.py`.
5. Build training batches for segmentation:  
`python referit/build_training_batches_seg.py`.
6. Configure the `config.py` file in the directory `det_model` and train the language-based bounding box localization model:  
`python referit/train_det_model.py`.
7. Configure the `config.py` file in the directory `seg_low_res_model` and train the low resolution language-based segmentation model (from the previous bounding box localization model):  
`python referit/train_low_res_model.py`.
8. Configure the `config.py` file in the directory `seg_model` and train the high resolution language-based segmentation model (from the previous low resolution segmentation model):  
`python referit/train_seg_model.py`.

### Evaluation
9. You may need to add the repository root directory to Python's module path:  
`export PYTHONPATH=path/to/text_objseg_caffe:$PYTHONPATH`.
10. Configure the `test_config.py` file in the directory `seg_model` and run evaluation for the high resolution language-based segmentation model:  
`python referit/test_seg_model.py`.  
This should reproduce the results in the paper.
11. You may also evaluate the language-based bounding box localization model:  
`python referit/test_det_model.py`.  
The results can be compared to [this paper](http://ronghanghu.com/text_obj_retrieval/).
