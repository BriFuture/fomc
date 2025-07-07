
## Prepare the dataset and run Experiments

### [DOTA](https://captain-whu.github.io/DOTA/dataset.html)

The DOTA dataset contains 15 classes. We select 10 classes as base classes and the remaining 5 classes as novel classes to train and validate several methods including FOMC(base detector: S2A-Net/ Oriented R-CNN), S2A-Net, R3Det, and Oriented RCNN.

1. Download DOTA dataset into `datasets/DOTA_v1` folder, decompress the zipped files, and rename the train and val folder into `trainA` and `valA`, respectively.

2. Split the imagesets into 1024x1024 crops following [here](fs/tools/)
```bash
python tools/img_split.py --base-json fs/tools/split_configs/ss_train.json
python tools/img_split.py --base-json fs/tools/split_configs/ss_val.json
```

The dataset folder structure should be as follows:

```
datasets/DOTA_v1/
├── trainA
│   ├── images
│   └── labelTxt
├── train_ss
│   ├── annfiles
│   ├── images
│   └── split
├── valA 
│   ├── images
│   └── labelTxt
└── val_ss
    ├── annfiles
    ├── images
    └── split
```

3. Base training stage: `python fs/train_and_eval_dota.py configs/dota/fomc/split1/ex_base.py --train`. 

4. Get the weights file and copy it into destination folder, prepare it for finetuning stage:

```bash
mkdir -p checkpoints/weights/dota/fomc/split1/base/
cp checkpoints/dota/fomc/split1/base/val_best_ckpt.pth checkpoints/weights/dota/fomc/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dota/fomc/split1/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/dota/fomc/split1/softmax --num-class=15 --method randinit
```

We provide the weights trained on the DOTA base categories on Google Drive. Please check [fomc_weights](https://drive.google.com/file/d/1oVdz2TIPbk473W2We4Fxgz37hBy3RVo6/view?usp=sharing). 

5. Identify each object for randomly selecting before finetuning stage. `python fs/scripts/dataset_id_objects/py --dataset=dota --config=configs/dota/ds_dota_origin.py`

6. Select shots and split the images into 1024x1024 crops. It should be noted that the option `--mask_unsel_shots` should be used if instance masking module is applied. Here the `--mask_unsel_shots` option is on for FOMC:

```bash
seed=5
shot=5

## with instance masking
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --mask_unsel_shots --dataset=dota --config=configs/dota//ds_dota_origin.py 
python fs/tools/novel_img_split.py --base-json fs/tools/split_configs/ss_train_split.json --seed ${seed} --shot ${shot} --mask

## without instance masking
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=dota --config=configs/dota/ds_dota_origin.py
python fs/tools/novel_img_split.py --base-json fs/tools/split_configs/ss_train_split.json --seed ${seed} --shot ${shot}
```

7. Fine-tune the model:

```bash
python fs/train_and_eval_dota.py configs/dota/fomc/split1/ex_shot10.py --train 
```

More detailed instructions are available in `fs/instructions/*.sh`.

-----


### [HRSC2016](https://www.kaggle.com/datasets/weiming97/hrsc2016-ms-dataset)

The training process of HRSC2016 is similar to that of DOTA. However, the object identification process can be skipped since the dataset has identified objects by `Object_ID` attribute. What's more, we train the dataset by resizing images into 1024x1024 instead of cropping them into 1024x1024 sub images. 

1. Download the dataset into `datasets/HRSC2016`, the dataset folder structure should be as follows:

```
datasets/HRSC2016
└── FullDataSet
    ├── AllImages
    ├── Annotations
    ├── ImageSets
    ├── LandMask
    ├── Segmentations
    ├── split
    └── sysdata.xml
```

2. Base train: `python fs/train_and_eval_hrsc.py configs/hrsc/fomc/split1/ex_base.py --train `

3. Copy it into destination folder, prepare it for finetuning stage: 

```bash
cp checkpoints/hrsc/fomc/split1/base/val_best_ckpt.pth checkpoints/weights/hrsc/fomc/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/hrsc/fomc/split1/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/hrsc/fomc/split1/softmax --num-class=20 --method randinit --prob_bias=0
```

4. Randomly select shots: 
```bash
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=hrsc --config=configs/hrsc/orcnn/split1/ds_orcnn_shot.py --mask_unsel_shots
```

5. Fine-tune the model: `python fs/train_and_eval_hrsc.py configs/hrsc/fomc/split1/ex_shot10.py --train `


-----

### [DIOR-R](https://gcheng-nwpu.github.io/)

The training process of DIOR-R is similar to that of DOTA.

1. Download the dataset into `datasets/DIOR-R`, the dataset folder structure should be as follows:

```
datasets/DIOR-R
├── All_Annotations
├── Annotations -> All_Annotations/Oriented Bounding Boxes
    ├── 00001.xml
    ├── .....
    └── 10000.xml
├── ImageSets
    ├── Layout
    ├── Main
    └── Segmentation
├── JPEGImages -> JPEGImages-trainval/
    ├── 00001.jpg
    ├── .....
    └── 10000.jpg
├── JPEGImages-test
├── JPEGImages-trainval
└── split
    ├── mask_seed5_shot5
    ├── .....
    └── seed5_shot5
```

1. Base train: `python fs/train_and_eval_dior.py configs/dior_r/fomc/split1/ex_base.py --train `

2. Copy it into destination folder, prepare it for finetuning stage: 
```bash
cp checkpoints/dior_r/fomc/split1/base/val_best_ckpt.pth checkpoints/weights/dior_r/fomc/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dior_r/fomc/split1/base/val_best_ckpt.pth --softmax --save-dir checkpoints/weights/dior_r/fomc/split1/softmax --num-class=20 --method randinit --prob_bias=0
```

3. Object identification is required since DIOR-R dataset does not provide `object-id` attribute: `python fs/scripts/dataset_id_objects.py --dataset=dior --config=configs/dior_r/ds_dior_origin.py`

4. Randomly select shots: 
```bash
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --mask_unsel_shots --dataset=dior --config=configs/dior_r/ds_dior.py 
```

5. Fine-tune the model:
```bash
python fs/train_and_eval_dior.py configs/dior_r/fomc/split1/ex_shot10.py --train 
```