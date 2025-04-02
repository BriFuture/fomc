#### ORCNN
## dota
mkdir -p checkpoints/weights/dota/r3det/split1/base/
cp checkpoints/dota/r3det/split1/base/val_best_ckpt.pth checkpoints/weights/dota/r3det/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/dota/r3det/split1/base/val_best_ckpt.pth   --save-dir checkpoints/weights/dota/r3det/split1/sigmoid --num-class=15 --method randinit --bbox_anchors=9

python fs/scripts/dataset_id_objects.py --dataset=dota --config=configs/dota/ds_dota_origin.py
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=dota --config=configs/dota/ds_dota_origin.py
python fs/tools/novel_img_split.py --base-json fs/tools/split_configs/ss_train_split.json --seed ${seed} --shot ${shot}
python fs/train_and_eval_dota.py configs/dota/r3det/split1/ex_shot${shot}.py --train 

splitid=2
mkdir -p checkpoints/weights/dota/r3det/split${splitid}/base/
cp checkpoints/dota/r3det/split${splitid}/base/val_best_ckpt.pth checkpoints/weights/dota/r3det/split${splitid}/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/dota/r3det/split${splitid}/base/val_best_ckpt.pth   --save-dir checkpoints/weights/dota/r3det/split${splitid}/sigmoid --num-class=15 --method randinit --bbox_anchors=9


## hrsc
mkdir -p checkpoints/weights/hrsc/r3det/split1/base/
cp checkpoints/hrsc/r3det/split1/base/val_best_ckpt.pth checkpoints/weights/hrsc/r3det/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/hrsc/r3det/split1/base/val_best_ckpt.pth   --save-dir checkpoints/weights/hrsc/r3det/split1/sigmoid --num-class=20 --method randinit --bbox_anchors=9

python fs/scripts/dataset_select_random_object.py --shot=10 --seed=6 --remove_exist_seed --dataset=hrsc --config=configs/hrsc/r3det/split1/ds_r3det_shot.py

## dior_r
mkdir -p checkpoints/weights/dior_r/r3det/split1/base/
cp checkpoints/dior_r/r3det/split1/base/val_best_ckpt.pth checkpoints/weights/dior_r/r3det/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/dior_r/r3det/split1/base/val_best_ckpt.pth  --save-dir checkpoints/weights/dior_r/r3det/split1/sigmoid --num-class=20 --method randinit --bbox_anchors=9

python fs/scripts/dataset_select_random_object.py --shot=10 --seed=6 --remove_exist_seed --dataset=dior --config=configs/dior_r/ds_dior.py
