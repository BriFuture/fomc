import tqdm
import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence
import logging
from collections import Counter

from bfcommon.fp16_utils import wrap_fp16_model
from gdet.data_factory import construct_labeled_dataset, construct_test_dataset

from gdet.datasets.evaluator.coco_metrics import TaskEvaluator
from gdet.registries import EXPS
from gdet.datasets.dataset import CocoDataset
from gdet.structures.configure import ConfigType
from gdet.structures.evaluation import ModelCocoOutput
from gdet.parallel.dist_utils import get_dist_info

from .exp_train import ExpTrain
logger = logging.getLogger("gdet.engine.train")

@EXPS.register_module()
class ExpCocoTrain(ExpTrain):
    def preprocess_config(self):
        config = self.m_config
        config.defrost()
        if config.model.use_base_category_indicator:
            config.model.base_category_indicator = config.dataset.base_category_indicator[:]
        config.freeze()
    
    def train_epoch(self, epoch: "int"):
        cfg = self.m_config
        train_cfg = cfg.train
        cfg_solver = train_cfg.solver
        total_iter = len(self.m_train_loader)
        fp16_cfg = train_cfg.fp16
        
        progress = enumerate(self.m_train_loader)
        self.m_dist_model.train()
        train_loss = []
        for iter, batch_data in progress:
            ## clear grad
            self.m_optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', enabled=fp16_cfg.enabled):
                output, loss_dicts = self.m_dist_model(batch_data)
            ## unfilter task
            loss_dicts: "dict[str, torch.Tensor]" 

            total_loss = sum(loss_dicts.values())
            total_loss_v = total_loss.item()
            train_loss.append(total_loss_v)
            if fp16_cfg.enabled:
                scaler = self.fp_scaler
                scaler.scale(total_loss).backward()
                self.dist_barrier()
                scaler.step(self.m_optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.dist_barrier()
                self.m_optimizer.step()

            total_epoch_iter = epoch * total_iter + iter
            self.m_iter = total_epoch_iter + 1
            self.warmup_lr(cfg_solver)

            if (iter + 1) % 10 == 0:
                loss_info = { k: np.round(v.item(), 5) for k, v in loss_dicts.items()}
                lr = self.m_optimizer.param_groups[0]['lr']
                logger.info(f"Train[{epoch}] {iter}/{total_iter}: Loss {total_loss_v:.5f}| {loss_info}, LR: {lr:.5f}")
        self.dist_barrier()

        if self.m_save_train_best:
            self.save_train_ckpt_interval(train_loss, epoch + 1)
        return train_loss
            
    
    def evaluate_val(self):
        if not self.m_eval_during_train:
            return
        ### make sure evaluation is constructed
        config = self.m_config
        if self.m_val_dataset is None:
            val_dataset, val_loader     = construct_labeled_dataset(config, "val")
            self.m_val_dataset = val_dataset
            self.m_val_loader = val_loader

        base_category_indicator = config.dataset.base_category_indicator
        
        ### bsf.c map indicator from label into coco catId 将 idx 和 name 映射起来
        mapped_base_indicator = self.m_val_dataset.get_mapped_indicator(base_category_indicator)
        if not self.m_train_mode:
            logger.info(f"Base Category: {base_category_indicator}")
            
            ### add fp16 feature
            fp16_cfg = config.val.fp16
            logger.info(f"config **FP16** enabled: {fp16_cfg.enabled}, train mode: {self.m_train_mode}")
        else:
            fp16_cfg = config.train.fp16
        

        self.class_ins_count = Counter()
        self.m_val_dataset: "CocoDataset"
        host_evaluator = TaskEvaluator(self.m_val_dataset.cfg.ann_file, per_category_metrics=True)
        self.m_dist_model.to(device=torch.device('cuda'))
        self.m_dist_model.eval()
        val_loss = []
        ###
        logger.info("Be aware that class idx is mapped into coco catId!!")
        label2catId = {v: k for k, v in self.m_val_dataset.m_catId2label.items()}

        total_data_len = len(self.m_val_loader)
        progress = tqdm.tqdm(enumerate(self.m_val_loader), total=total_data_len, desc='Evaluating')
        ### 将模型输出的 idx 映射到 coco id
        map_tensor = torch.zeros(len(label2catId)+1, dtype=torch.float32).to(self.device)  # 0-10 range
        for k, v in label2catId.items():
            map_tensor[k] = v
        with torch.no_grad():
            for iter, batch_data in progress:
                with torch.autocast(device_type='cuda', enabled=fp16_cfg.enabled):
                    output: "ModelCocoOutput" = self.m_dist_model(batch_data)

                det_classes: "torch.Tensor" = output['detection']['detection_classes']
                ### 将 empty 输出设置成 0
                mask = det_classes >= len(map_tensor)
                det_classes[mask] = 0
                mapped_classes = map_tensor[det_classes.long()]
                output['detection']['detection_classes'] = mapped_classes
                host_evaluator.remove_unnecessary_keys(output)
                ### map detection results into original coco id
                host_evaluator.task_update(output)
                # print("val", iter,  total_data_len)
        
        if config.dist.rank != 0:
            pass

        self.dist_barrier()
        ### TODO; if rank is not 0, recv other information, else store information into /dev/shm/torch_dist/rank_{i}
        if config.dist.rank == 0:
            pass

        print("RPN match count" , self.m_model.match_novel_count)
        # summary_writer = tensorboard.SummaryWriter(log_dir) 
        
        output_metrics: "dict" = host_evaluator.task_evaluate(mapped_base_indicator)
        self.save_val_ckpt(output_metrics)
        # metrics.write_dict_metrics(0, output_metrics, summary_writer)
        # metric_path = osp.join(log_dir, "detection_metrics.json")
        # logging.info("Writing to metric path %s", metric_path)
        if len(val_loss):
            val_loss_v = np.mean(val_loss)
            print("Val loss: ", val_loss_v)
        ### bsf.c here stats by label, not coco catId
        stat_ins = {}
        for k in sorted(self.class_ins_count.keys()):
            stat_ins[k] = self.class_ins_count[k]
        print("Val object stats :", stat_ins)
