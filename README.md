# FCOS: Fully Convolutional One-Stage Object Detection

This project hosts the code for implementing the semi-supervised object detection on the top of FCOS and Mask-rcnn. The general pipeline is 
1) run inference on the partial training dataset and extract features (optional)
2) generate pseudo labels
3) re-train/finetune the model

We also have additional tools for evaluation and visualization of the generated pseudo labels.

Please refer to the original FCOS [README.md](FCOS_README.md) for more details of this detection model.

## Installation 
This FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## Dataset preparation
To start with the semi-supervised learning, we need a partially labeled training set and its corresponding fully labeled one for evaluation.
```
cp partial_datasets/*.json datasets/coco/annotations/
```

## Pipeline
Suppose we've already had a model trained on the partial training datasets (the stage1 model), then we need to generate pseudo labels from this model and re-train/finetune this model.

### Run inference
Directly run the script `extract_feature.sh` or run 
```
python tools/feature_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x_for_feature_extraction.yaml \
    MODEL.WEIGHT [.pth file] \
    OUTPUT_DIR [outout directory]
``` 

### Generate pseudo labels
```
python tools/generate_pseudo_label.py \
    --predictions [directory of predictions same as above] \
    --annotation [output annotation json file path]
```

### Evaluate pseudo labels
```
python tools/compute_pseudo_label_quality.py \
    --annotation [annotation json file path]
```

### Re-train
Directly run the script `train.sh`(single GPU version) or `train_2gpu.sh` (multiple GPU version).
Note that, if you want to use a soft coefficient for the pseudo loss term, you can assign `MODEL.PSEUDO_WEIGHT` in the config file (usuallt we use `configs/fcos/fcos_imprv_R_50_FPN_1x_pseudo.yaml`).

### Miscellaneous

#### Visualization tools
1) `visualize_log.py`: visualize loss curves along with different stages.
2) `visualize_label_quality.py`: scatter plot of IoU vs nearest neighbor distance. It requires results from `NN_query.py`.
2) `visualize_pseudo_label.py`: visualize pseudo labels, partial labels and missing labels on the images.

#### Nearest Neighbor query
This part requires to install the [faiss](https://github.com/facebookresearch/faiss) library.