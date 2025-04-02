import os, os.path as osp
import tqdm
import logging
import numpy as np
import torch
import torch.distributed as dist
from collections import Counter
import shutil as sh

from gdet.data_factory import construct_labeled_dataset, construct_test_dataset
from gdet.registries import EXPS
from .exp_eval import ExpEval
from .initial import initialize_optimizer
from .env import ctx_mgr

logger = logging.getLogger("gdet.engine.train")


@EXPS.register_module()
class ExpTrain(ExpEval):
    val_metric_key = "AP50"
    def init(self):
        super().init()
        self.m_val_ap = 0
        
        self.m_best_loss = 10000
        config = self.m_config
        self.m_log_train_interval = config.train.get("log_interval", 10)
        self.m_use_cache_detection_result = config.experiment.get("use_cache_detection_result", False)
        
    @property
    def m_epoch(self):
        return ctx_mgr.get_epoch()

    @m_epoch.setter
    def m_epoch(self, value):
        ctx_mgr.set_epoch(value)

    @property
    def m_iter(self):
        return ctx_mgr.get_iter()

    @m_iter.setter
    def m_iter(self, value):
        ctx_mgr.set_iter(value)

    def load_model_state(self):
        """从 saved_state_dict 中加载，其中包含 epoch, 模型metric 等数据
        """
        load_from = self.m_config.experiment.load_from
        if osp.exists(load_from):
            saved_dict = torch.load(load_from, map_location="cpu")
            sd = saved_dict['state_dict']
            incompat_keys = None
            try:
                incompat_keys = self.m_model.load_state_dict(sd, strict=False)
            except Exception as e:
                logger.warning(e)
            # incompat_keys = self.m_model.load_state_dict(sd, strict=True)
            logger.info(f"Load From : {load_from}")
            logger.info(f"Incompatible keys: {incompat_keys}")
        elif load_from:
            logger.warning(f"Unable to find weight: {load_from}")
    
    def auto_copy_best_ckpt(self, base_name):
        """自动保存上一次的最好的ckpt结果
        """
        wd = self.m_config.experiment.work_dir
        i = ""
        name = f"{base_name}"
        dst = osp.join(wd, name)
        if osp.exists(dst):
            for i in range(10):
                iname = f"{base_name}{i}"
                idst = osp.join(wd, iname)
                if not osp.exists(idst):
                    break
            else:
                idst = None
            if idst is not None:
                sh.copy(dst, idst)
                logger.info(f"Copy last best checkpoint {dst} into {idst}")
            else:
                print(f"Please checkout all checkpoints and continue: {base_name}")
                raise ValueError(f"Checkpoints should be checked: {wd}")

    def train(self):
        config = self.m_config
        if config.train.auto_copy_best:
            self.auto_copy_best_ckpt("train_best_ckpt.pth")
            self.auto_copy_best_ckpt("val_best_ckpt.pth")
        self.save_checkpoints = config.train.save_checkpoints
        self.eval_checkpoints = config.train.eval_checkpoints
        self.m_train_mode = True

        train_dataset, train_loader = construct_labeled_dataset(config, "train")
        self.m_train_dataset = train_dataset
        self.m_train_loader = train_loader
        rank = config.dist.rank

        if rank == 0:
            val_dataset, val_loader = construct_labeled_dataset(config, "val")
            self.m_val_dataset = val_dataset
            self.m_val_loader = val_loader

        ### check class count
        self.class_ins_count = Counter()
        self.m_optimizer = initialize_optimizer(config, self.m_dist_model)
        self.m_dist_model.train()
        self.m_model.construct_loss()
        
        cfg_solver = config.train.solver
        ### add fp16 feature
        fp16_cfg = config.train.fp16
        if fp16_cfg.enabled:
            self.fp_scaler = torch.cuda.amp.GradScaler(init_scale=fp16_cfg.loss_scale)
        
        max_epochs = config.experiment.epochs
        for epoch in range(max_epochs):
            self.m_epoch = epoch
            train_loss = self.train_epoch(epoch)
            
            self.update_lr_by_epoch(epoch, cfg_solver)

            if rank == 0 and self.eval_checkpoints > 0 and epoch % self.eval_checkpoints == 0:
                self.evaluate_val()
        
        self.save_train_ckpt_interval(train_loss, max_epochs, force=True)
        ### bsf.c 增加last epoch evalu
        if rank == 0 and self.eval_checkpoints > 0:
            self.evaluate_val()
            
        stat_ins = {}
        for k in sorted(self.class_ins_count.keys()):
            stat_ins[k] = self.class_ins_count[k]
        print("Train Stats count", stat_ins)

    def warmup_lr(self, cfg_solver):
        """根据 iter 进行 learning rate 的 warm up strategy
        """
        total_epoch_iter = self.m_iter
        if total_epoch_iter <= cfg_solver.warmup_steps:
            curr_lr = (total_epoch_iter) / cfg_solver.warmup_steps * (cfg_solver.learning_rate) 
            for pg in self.m_optimizer.param_groups:
                pg['lr'] = curr_lr

    def update_lr_by_epoch(self, epoch, cfg_solver):
        if epoch in cfg_solver.decay_epochs:
            curr_lr = list(self.m_optimizer.param_groups)[0]['lr']
            curr_lr *= cfg_solver.decay_rate
            for pg in self.m_optimizer.param_groups:
                pg['lr'] = curr_lr

    def get_curr_lr(self):
        curr_lr = list(self.m_optimizer.param_groups)[0]['lr']
        return curr_lr

    def train_epoch(self, epoch: "int"):
        cfg = self.m_config
        cfg_solver = cfg.train.solver
        total_iter = len(self.m_train_loader)
        
        progress = enumerate(self.m_train_loader)
        self.m_dist_model.train()
        train_loss = []
        for iter, batch_data in progress:
            ## clear grad
            self.m_optimizer.zero_grad()

            output, loss_dicts = self.m_dist_model(batch_data)
            ## unfilter task
            loss_dicts: "dict[str, torch.Tensor]" # = self.m_model.loss(self.m_loss_func, output, batch_data)

            total_loss = sum(loss_dicts.values())
            total_loss_v = total_loss.item()
            train_loss.append(total_loss_v)

            self.m_optimizer.zero_grad()
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
            self.save_train_ckpt_interval(train_loss, epoch+1)
        return train_loss

    def save_train_ckpt_interval(self, train_loss: "np.ndarray", epoch, force = False):
        cfg = self.m_config
        if cfg.dist.rank != 0:
            return
        try:
            train_loss_v = np.mean(train_loss)
        except:
            np.save("/dev/shm/train_ckpt_loss_err.dat", train_loss)
            train_loss_v = 0
        state_dict = self.m_model.state_dict()
        wd = cfg.experiment.work_dir
        if train_loss_v < self.m_best_loss and False:
            self.m_best_loss = train_loss_v
            
            save_dict = {
                "loss": self.m_best_loss,
                "state_dict": state_dict,
                "epoch": epoch,
            }
            dst = osp.join(wd, "train_best_ckpt.pth")
            torch.save(save_dict, dst)
            logger.info(f"Save Best Train Checkpoints: {epoch}, {dst}")
        if (self.save_checkpoints > 0 and epoch % self.save_checkpoints == 0) or force:
            save_dict = {
                "loss": train_loss_v,
                "state_dict": state_dict,
                "epoch": epoch,
            }
            dst = osp.join(wd, f"train_ckpt_ep{epoch}.pth")
            torch.save(save_dict, dst)
            force_str = "force" if force else ""
            logger.info(f"Save Train Checkpoints ({force_str}): {epoch}, {dst}")
            save_dict = {
                "optimizer": self.m_optimizer.state_dict(),
                "epoch": epoch
            }
            dst = osp.join(wd, f"train_optimizer_ep{epoch}.pth")
            torch.save(save_dict, dst)
            logger.info(f"Save Train Optimizer: {epoch}, {dst}")

    def save_val_ckpt(self, output_metrics: "dict"):
        cfg = self.m_config
        if cfg.dist.rank != 0:
            return
        wd = cfg.experiment.work_dir
        if not self.m_train_mode:
            return
        ### eval during training
        metric_log = osp.join(wd, "metrics.log")
        with open(metric_log, "a") as f:
            f.write(f"-->Epoch: {self.m_epoch}\n")
            for k, v in output_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")

        if self.m_save_val_best:
            ### save by metric
            ap50 = output_metrics[self.val_metric_key]
            if ap50 > self.m_val_ap:
                logger.info(f"Validation ap increased: {self.m_val_ap} ---> {ap50}")
                self.m_val_ap = ap50
                state_dict = self.m_model.state_dict()
                save_dict = {
                    "loss": self.m_best_loss,
                    "metrics": output_metrics,
                    "state_dict": state_dict,
                    "epoch": self.m_epoch,
                }
                dst = osp.join(wd, "val_best_ckpt.pth")
                torch.save(save_dict, dst)
                logger.info(f"Save Best val Checkpoints: {self.m_epoch}, {dst}")

    def evaluate_val(self):
        ### make sure evaluation is constructed
        raise ValueError("please use ExpCocoTrain for default training")


    def evaluate_test(self):
        """unimplemented
        """
        cfg = self.m_config
        test_dataset, test_loader = construct_test_dataset(cfg)
        total_iter = len(test_loader)
        for iter, batch_data in test_loader:
            for k in list(batch_data.keys()):
                batch_data[k] = batch_data[k].data[0]
            output = self.m_model(batch_data)


    def dist_barrier(self):
        if self.m_config.dist.distributed:
            dist.barrier()
        