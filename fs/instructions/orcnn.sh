#### ORCNN
## dota
mkdir -p checkpoints/weights/dota/orcnn/split1/base/
cp checkpoints/dota/orcnn/split1/base/val_best_ckpt.pth checkpoints/weights/dota/orcnn/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dota/orcnn/split1/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/dota/orcnn/split1/softmax --num-class=15 --method randinit

python fs/scripts/dataset_id_objects.py --dataset=dota --config=configs/dota/ds_dota_origin.py
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=dota --config=configs/dota/ds_dota_origin.py
python fs/tools/novel_img_split.py --base-json fs/tools/split_configs/ss_train_split.json --seed ${seed} --shot ${shot}
python fs/train_and_eval_dota.py configs/dota/orcnn/split1/ex_shot${shot}.py --train 

splitid=2
mkdir -p checkpoints/weights/dota/orcnn/split${splitid}/base/
cp checkpoints/dota/orcnn/split${splitid}/base/val_best_ckpt.pth checkpoints/weights/dota/orcnn/split${splitid}/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dota/orcnn/split${splitid}/base/val_best_ckpt.pth --softmax    --save-dir checkpoints/weights/dota/orcnn/split${splitid}/softmax --num-class=15 --method randinit

## hrsc
cp checkpoints/hrsc/orcnn/split1/base/val_best_ckpt.pth checkpoints/weights/hrsc/orcnn/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/hrsc/orcnn/split1/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/hrsc/orcnn/split1/softmax --num-class=20 --method randinit

python fs/scripts/dataset_select_random_object.py --shot=10 --seed=6 --remove_exist_seed --dataset=hrsc --config=configs/hrsc/orcnn/split1/ds_orcnn_shot.py

## dior_r
cp checkpoints/dota/orcnn/split1/base/val_best_ckpt.pth checkpoints/weights/dota/orcnn/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py checkpoints/weights/dior_r/orcnn/split1/base/val_best_ckpt.pth --softmax --save-dir checkpoints/weights/dior_r/orcnn/split1/softmax --num-class=20 --method randinit

python fs/scripts/dataset_select_random_object.py --shot=10 --seed=6 --remove_exist_seed --dataset=dior --config=configs/dior_r/ds_dior.py
