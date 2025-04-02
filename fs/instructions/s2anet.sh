
#### S2ANet
## dota
python fs/train_and_eval_dota.py configs/dota/s2anet/split1/ex_base.py --train 

mkdir -p checkpoints/weights/dota/s2anet/split1/base/
cp checkpoints/dota/s2anet/split1/base/val_best_ckpt.pth checkpoints/weights/dota/s2anet/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/dota/s2anet/split1/base/val_best_ckpt.pth  --save-dir checkpoints/weights/dota/s2anet/split1/sigmoid --num-class=15 --method randinit
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=dota --config=configs/dota//ds_dota_origin.py
python fs/tools/novel_img_split.py --base-json fs/tools/split_configs/ss_train_split.json --seed ${seed} --shot ${shot}
python fs/train_and_eval_dota.py configs/dota/s2anet/split1/ex_shot3.py --train 

splitid=2
mkdir -p checkpoints/weights/dota/s2anet/split${splitid}/base/
cp checkpoints/dota/s2anet/split${splitid}/base/val_best_ckpt.pth checkpoints/weights/dota/s2anet/split${splitid}/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/dota/s2anet/split${splitid}/base/val_best_ckpt.pth   --save-dir checkpoints/weights/dota/s2anet/split${splitid}/sigmoid --num-class=15 --method randinit

## hrsc
cp checkpoints/hrsc/s2anet/split1/base/val_best_ckpt.pth checkpoints/weights/hrsc/s2anet/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py --src1 checkpoints/weights/hrsc/s2anet/split1/base/val_best_ckpt.pth  --save-dir checkpoints/weights/hrsc/s2anet/split1/sigmoid --num-class=20 --method randinit

python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=hrsc --config=configs/hrsc/orcnn/split1/ds_orcnn_shot.py

## dior_r
cp checkpoints/dior/s2anet/split1/base/val_best_ckpt.pth checkpoints/weights/dior/s2anet/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_sigmoid.py checkpoints/weights/dior_r/s2anet/split1/base/val_best_ckpt.pth --save-dir checkpoints/weights/dior_r/s2anet/split1/sigmoid --num-class=20 --method randinit

python fs/scripts/dataset_id_objects.py --dataset=dior --config=configs/dior_r/ds_dior_origin.py
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=dior --config=configs/dior_r/ds_dior.py
