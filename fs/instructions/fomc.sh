#### FOMC
## dota 
#========== split1
python fs/train_and_eval_dota.py configs/dota/fomc/split1/ex_base.py --train 
mkdir -p checkpoints/weights/dota/fomc/split1/base/
cp checkpoints/dota/fomc/split1/base/val_best_ckpt.pth checkpoints/weights/dota/fomc/split1/base/val_best_ckpt.pth

python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dota/fomc/split1/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/dota/fomc/split1/softmax --num-class=15 --method randinit
python fs/scripts/dataset_id_objects.py --dataset=dota --config=configs/dota/ds_dota_origin.py
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --mask_unsel_shots --dataset=dota --config=configs/dota//ds_dota_origin.py 
python fs/tools/novel_img_split.py --base-json fs/tools/split_configs/ss_train_split.json --seed ${seed} --shot ${shot} --mask
python fs/train_and_eval_dota.py configs/dota/fomc/split1/ex_shot10.py --train 

splitid=2
mkdir -p checkpoints/weights/dota/fomc/split${splitid}/base/
cp checkpoints/dota/fomc/split${splitid}/base/val_best_ckpt.pth checkpoints/weights/dota/fomc/split${splitid}/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dota/fomc/split${splitid}/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/dota/fomc/split${splitid}/softmax --num-class=15 --method randinit --prob_bias=0


#========== split2
cp checkpoints/dota/fomc/split${splitid}/base/val_best_ckpt.pth checkpoints/weights/dota/fomc/split${splitid}/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dota/fomc/split${splitid}/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/dota/fomc/split1/softmax --num-class=15 --method randinit --prob_bias=0


## hrsc 数据集
cp checkpoints/hrsc/fomc/split1/base/val_best_ckpt.pth checkpoints/weights/hrsc/fomc/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/hrsc/fomc/split1/base/val_best_ckpt.pth  --softmax --save-dir checkpoints/weights/hrsc/fomc/split1/softmax --num-class=20 --method randinit --prob_bias=0
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --dataset=hrsc --config=configs/hrsc/orcnn/split1/ds_orcnn_shot.py --mask_unsel_shots
python fs/train_and_eval_hrsc.py configs/hrsc/fomc/split1/ex_shot10.py --train 


## dior_r
cp checkpoints/dior_r/fomc/split1/base/val_best_ckpt.pth checkpoints/weights/dior_r/fomc/split1/base/val_best_ckpt.pth
python fs/scripts/ckpt_surgery_softmax.py --src1 checkpoints/weights/dior_r/fomc/split1/base/val_best_ckpt.pth --softmax --save-dir checkpoints/weights/dior_r/fomc/split1/softmax --num-class=20 --method randinit --prob_bias=0

python fs/scripts/dataset_id_objects.py --dataset=dior --config=configs/dior_r/ds_dior_origin.py
seed=5
shot=5
python fs/scripts/dataset_select_random_object.py --shot=${shot} --seed=${seed} --remove_exist_seed --mask_unsel_shots --dataset=dior --config=configs/dior_r/ds_dior.py 
python fs/train_and_eval_dior.py configs/dior_r/fomc/split1/ex_shot10.py --train 
