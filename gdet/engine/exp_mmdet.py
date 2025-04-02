import tqdm
import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Sequence
from collections import Counter
from gdet.structures.configure import TrainConfigType
from gdet.data_factory import construct_labeled_dataset, construct_test_dataset
from gdet.registries import EXPS, EVALUATORS
from .initial import initialize_optimizer
from .exp_train import ExpTrain

logger = logging.getLogger("gdet.engine.fs")


@EXPS.register_module()
class ExpMMDetTrain(ExpTrain):
    val_metric_key = "mAP"
    def preprocess_config(self):
        config = self.m_config
        config.defrost()
        if config.model.use_base_category_indicator:
            config.model.base_category_indicator = config.dataset.base_category_indicator[:]
        config.freeze()
    
    def train(self):
        config = self.m_config
        train_cfg: "TrainConfigType" = config.train
        if train_cfg.auto_copy_best:
            self.auto_copy_best_ckpt("train_best_ckpt.pth")
            self.auto_copy_best_ckpt("val_best_ckpt.pth")
        self.save_checkpoints = train_cfg.save_checkpoints
        self.eval_start = train_cfg.eval_start
        self.eval_checkpoints = train_cfg.eval_checkpoints
        self.m_train_mode = True

        train_dataset, train_loader = construct_labeled_dataset(config, "train")
        self.m_train_dataset = train_dataset
        self.m_train_loader = train_loader

        ### check class count
        self.class_ins_count = Counter()
        self.m_optimizer = initialize_optimizer(config, self.m_dist_model)
        self.m_dist_model.train()
        # self.m_model.construct_loss()
        
        cfg_solver = train_cfg.solver
        ### add fp16 feature
        fp16_cfg = train_cfg.fp16
        if fp16_cfg.enabled:
            self.fp_scaler = torch.cuda.amp.GradScaler(init_scale=fp16_cfg.loss_scale)
            logger.info(f"Train process enable fp16: {fp16_cfg.enabled}")
        
        max_epochs = config.train.solver.total_epochs
        self.m_max_epoch = max_epochs
        for epoch in range(max_epochs):
            self.m_epoch = epoch
            train_loss = self.train_epoch(epoch)
            
            self.update_lr_by_epoch(epoch, cfg_solver)

            if epoch >= self.eval_start and self.eval_checkpoints > 0 and ((epoch + 1) % self.eval_checkpoints) == 0:
                self.evaluate_val()
        ### bsf.c 增加last epoch evalu
        self.save_train_ckpt_interval(train_loss, max_epochs, force=True)
        if self.eval_checkpoints > 0:
            self.evaluate_val()
            
        stat_ins = {}
        for k in sorted(self.class_ins_count.keys()):
            stat_ins[k] = self.class_ins_count[k]
        print("Train Stats count", stat_ins)

    def train_epoch(self, epoch: "int"):
        cfg = self.m_config
        cfg_solver = cfg.train.solver
        optimizer_config = cfg_solver.optimizer_config
        grad_clip_cfg = optimizer_config.get("grad_clip", None)
        total_iter = len(self.m_train_loader)
        fp16_cfg = cfg.train.fp16
        
        progress = enumerate(self.m_train_loader)
        self.m_dist_model.train()
        train_loss = []
        rank = cfg.dist.rank
        self.dist_barrier()
        has_logged = False
        for iter, batch_data in progress:
            
            ## clear grad
            self.m_optimizer.zero_grad()
            fp16_en = fp16_cfg.enabled and fp16_cfg.start_iter < self.m_iter
            with torch.autocast(device_type='cuda', enabled=fp16_en):
                output = self.m_dist_model(**batch_data)
                loss_dicts = {}
                for k, v in output.items():
                    assert not torch.isnan(torch.as_tensor(v)).any()
                    if isinstance(v, Sequence):
                        loss_dicts[k] = sum(v)
                    elif isinstance(v, torch.Tensor):
                        loss_dicts[k] = v
            ## unfilter task
            loss_dicts: "dict[str, torch.Tensor]"
            total_loss = sum(loss_dicts.values())
            total_loss_v = total_loss.item()

            train_loss.append(total_loss_v)
            if fp16_en:
                scaler = self.fp_scaler
                scaler.scale(total_loss).backward()

                if grad_clip_cfg is not None:
                    scaler.unscale_(self.m_optimizer)
                    nn.utils.clip_grad_norm_(self.m_model.parameters(), grad_clip_cfg.max_norm, grad_clip_cfg.norm_type)
                
                self.dist_barrier()

                scaler.step(self.m_optimizer)
                scaler.update()
            else:
                total_loss.backward()
   
                if grad_clip_cfg is not None:
                    nn.utils.clip_grad_norm_(self.m_model.parameters(), grad_clip_cfg.max_norm, grad_clip_cfg.norm_type)
                self.dist_barrier()
                self.m_optimizer.step()
                
            total_epoch_iter = epoch * total_iter + iter
            self.m_iter = total_epoch_iter + 1
            self.warmup_lr(cfg_solver)

            if (iter + 1) % self.m_log_train_interval == 0:
                loss_info = { k: np.round(v.item(), 5) for k, v in loss_dicts.items()}
                lr = self.get_curr_lr()
                logger.info(f"Train[{epoch}/{self.m_max_epoch}] {iter + 1}/{total_iter}: Loss {total_loss_v:.5f}| {loss_info}, LR: {lr:.5f}")
                has_logged = True

        if not has_logged:
            loss_info = { k: np.round(v.item(), 5) for k, v in loss_dicts.items()}
            lr = self.get_curr_lr()
            logger.info(f"Train[{epoch}/{self.m_max_epoch}] {iter + 1}/{total_iter}: Loss {total_loss_v:.5f}| {loss_info}, LR: {lr:.5f}")

        if self.m_save_train_best:
            self.save_train_ckpt_interval(train_loss, epoch+1)
            
        self.dist_barrier()
        return train_loss
    
    def evaluate_val(self, ds_name="val"):
        if not self.m_eval_during_train:
            return
        ### make sure evaluation is constructed
        config = self.m_config
        ## if ds_name == "val":
        if self.m_val_dataset is None:
            val_dataset, val_loader     = construct_labeled_dataset(config, ds_name)
            self.m_val_dataset = val_dataset
            self.m_val_loader = val_loader
        self._evaluate(self.m_val_dataset, self.m_val_loader, ds_name)

    def _evaluate(self, val_dataset, val_loader, ds_name):
        config = self.m_config
        if not self.m_train_mode:
            ### add fp16 feature
            fp16_cfg = config[ds_name].fp16
            fp16_en = fp16_cfg.enabled
            logger.info(f"val config **FP16** enabled: {fp16_en}")
        else:
            fp16_cfg = config.train.fp16
            fp16_en = fp16_cfg.enabled or config[ds_name].fp16.enabled
            logger.info(f"val during train config **FP16** enabled: {fp16_en}")

        self.class_ins_count = Counter()
        
        evaluator_cfg = config.dataset.evaluator.clone()
        evaluator_cls_type = evaluator_cfg.pop("type")
        evaluator_cls = EVALUATORS.get(evaluator_cls_type)
        assert evaluator_cls is not None, evaluator_cls_type
        host_evaluator = evaluator_cls(evaluator_cfg, val_dataset)

        self.m_dist_model.to(device=torch.device('cuda'))
        self.m_dist_model.eval()
        ###
                
        total_data_len = len(val_loader)
        rank_id = config.dist.rank
        if rank_id == 0:
            progress = tqdm.tqdm(enumerate(val_loader), total=total_data_len, desc='Evaluating')
        else:
            progress = enumerate(val_loader)
        
        with torch.no_grad():
            cache_file = f"/dev/shm/cached_mmdet_results_rank{rank_id}.dat"
            import pickle
            final_output = None
            if osp.exists(cache_file) and self.m_use_cache_detection_result:
                with open(cache_file, "rb") as f:
                    cached_results = pickle.load(f)
                    if cached_results['load_from'] == self.m_config.experiment.load_from:
                        final_output = cached_results['output']

            if final_output is None:
                final_output = {}


                for iter, batch_data in progress:
                    with torch.autocast(device_type='cuda', enabled=fp16_en):
                        batch_data['return_loss'] = False
                        batch_data['rescale'] = True
                        batch_data.pop("ids", None)
                        output: "list[list]" = self.m_dist_model(**batch_data)
                    img_names =[ im['filename'] for im in batch_data['img_metas'][0].data[0] ]
                    for im_loc, det_out in zip(img_names, output):
                        iname = osp.basename(im_loc)
                        assert iname not in final_output
                        final_output[iname] = det_out

                if self.m_use_cache_detection_result:
                    with open(cache_file, "wb") as f:
                        cached_results = {
                            "output": final_output,
                            "load_from": self.m_config.experiment.load_from,
                        }
                        pickle.dump(cached_results, f)
        world_size = config.dist.world_size  ### 默认是 1 
        if world_size > 1:
            root_dir = "/dev/shm/exp/dist_train_results"
            os.makedirs(root_dir, exist_ok=True)
            saved_result_format = osp.join(root_dir, f"saved_results_{rank_id}.dat")
            if rank_id != 0:
                with open(saved_result_format, "wb") as f:
                    cached_results = {
                        "output": final_output,
                        "load_from": self.m_config.experiment.load_from,
                        "rank": rank_id
                    }
                    pickle.dump(cached_results, f)
                # print(f"rank: {rank_id}, {len(final_output)}, saved into: {saved_result_format}", )
            self.dist_barrier()
            ### bsf.c if rank is not 0, recv other information, else store information into /dev/shm/torch_dist/rank_{i}
            if rank_id != 0:
                return

            for r_idx in range(1, world_size):
                saved_result_format = osp.join(root_dir, f"saved_results_{r_idx}.dat")
                with open(saved_result_format, "rb") as f:
                    cached_results = pickle.load(f)
                    
                    rank_output = cached_results['output']
                # rank_outputs.append(rank_output)
                final_output.update(rank_output)
                ## remove it
                os.remove(saved_result_format)
            ## 重新设置 final_output 顺序
            new_final_output = {}
            for i in range(len(val_dataset)):
                di = val_dataset.m_data_infos[i]
                filename = osp.basename(di['filename'])
                new_final_output[filename] = final_output[filename]
            # print(f"rank: {rank_id}, {len(final_output)} {len(val_dataset)}", )
            final_output = new_final_output

        host_evaluator.task_update(final_output)
        output_metrics: "dict" = host_evaluator.evaluate(final_output)
        self.save_val_ckpt(output_metrics)

    def calc_flops(self):
        from mmengine.analysis import get_model_complexity_info
        config = self.m_config
        model = self.m_model
        model.eval()
        ds_name = "val"
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        if self.m_val_dataset is None:
            val_dataset, val_loader     = construct_labeled_dataset(config, ds_name)
            self.m_val_dataset = val_dataset
            self.m_val_loader = val_loader
        for iter, batch_data in enumerate(val_loader):
            batch_data['return_loss'] = False
            _, kwargs = self.m_dist_model.scatter(tuple(), batch_data, [0])
            batch_data = kwargs[0]
            img = batch_data['img'][0]
            img_metas = batch_data['img_metas'][0]
            results = get_model_complexity_info(model, inputs=(img[:1, ...],))
            break

        flops = results['flops']
        params = results['params']
        flops_str = results['flops_str']
        params_str = results['params_str']
        print(f"Total FLOPs: {flops / 1e9:.2f} G, original: {flops_str}")  # GFLOPs
        print(f"Total Params: {params / 1e6:.2f} M, original: {params_str}")  # M params