# Ped_VLM.


## Installation
1. Clone this repository
2. In the repository directory, run `mkdir results`
3. To replicate our environment use the `env.yml` we have provided. The following commands should create a proper environment:

```
conda env create -f env.yml
conda activate ped_VLM
```

## Dataset
The folder structure should now be as follows: 
```
└── rootFolder
  ├── Ped_Dataset/
      ├── train.json
      ├── val.json
      ├── test.json
      ├── test_coco.json
      ├── image_id.json
  ├── data/
    ├── JAAD/opticalflow
    ├── JAAD/images
    ├── PIE/opticalflow
    ├── PIE/images
    ├── data/Titan/images_anonymized/clip_xx/opticalflow/
    ├── data/Titan/images_anonymized/clip_xx/images/
```
 
## Training
* To run training, run `python train.py --batch-size [BATCH SIZE] --epochs [EPOCHS] --lm {T5-Base, T5-Large} `. For more information on other hyperparameters such as loading checkpoints or altering learning rate, weight decay, or the hidden size for gated pooling attention, run `python train.py --help`.
