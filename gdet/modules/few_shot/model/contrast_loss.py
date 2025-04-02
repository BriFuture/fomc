import torch
import torch.nn as nn
import torch.nn.functional as F
from . import weight_init
from mmcv.cnn import normal_init
import logging

logger = logging.getLogger("gdet.model.con_loss")

class ConvContrastiveHead(nn.Module):
    """Conv head for contrastive representation learning, 
    https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in = 256, feat_dim = 1024, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.layer1 = nn.Conv2d(dim_in, feat_dim, 3, padding=1)
        self.layer2 = nn.Conv2d(feat_dim, dim_out, 3, padding=1)
        head = nn.Sequential(
            self.layer1,
            self.layer2,
            
        )
        self.disable_fp16 = False
        for layer in head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
            elif isinstance(layer, nn.Conv2d):
                normal_init(layer)

    def forward(self, x):
        # feat = self.head(x)
        feat = self.layer1(x)
        feat = F.relu(feat, inplace=True)
        feat = self.layer2(feat)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


class LinearContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in = 256, feat_dim = 1024, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim_out),
        )
        self.disable_fp16 = True
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
    def half(self):
        print("Disable Half for contrast")
        return self
        
    def forward(self, x):
        # feat = self.head(x)
        # feat_normalized = F.normalize(feat, dim=1)
        # return feat_normalized
        return self.head(x)

class PredictorContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in = 256, feat_dim = 1024, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim, bias=True),
            # nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim_out, bias=True),
        )
        self.disable_fp16 = True
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
            
    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized
        # return self.head(x)

ContrastiveHead = LinearContrastiveHead
class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none', **kwargs):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_func)
        self.disable_fp16 = True

    def forward(self, features, labels, ious, *args):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        keep = (ious >= self.iou_threshold) 
        labels = labels[keep]
        features = F.normalize(features[keep], dim=1)
        ious = ious[keep]
        if features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(torch.matmul(features, features.T), self.temperature) # T[512, 512]
        
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0) # 去除自身的相似

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_(1e-5))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        loss = -per_label_log_prob

        coef = self.reweight_func(ious)
        # coef = coef[keep]

        loss = loss * coef
        return loss.mean()
        

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)

        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou
        def log_decay(iou):
            return torch.log2(iou + 1)

        if option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay
        elif option == 'log':
            return log_decay
        else:
            return trivial

class WeightedSupConLoss(SupConLoss):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, features, labels, ious, probs=None):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        keep = (ious >= self.iou_threshold) 
        labels = labels[keep]
        features = F.normalize(features[keep], dim=1)
        ious = ious[keep]

        if features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(torch.matmul(features, features.T), self.temperature) # T[512, 512]
        
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0) # 去除自身的相似

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_(1e-5))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
       
        loss = -per_label_log_prob
        coef = self.reweight_func(ious)
        loss = loss * coef

        return loss.mean()

import math, os.path as osp
class BankSupConLoss(SupConLoss):
    """Supervised Contrastive LOSS as defined in """
    def __init__(self,  queue_size=8192, dim=256, base_feature=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_size = queue_size
        self.register_buffer( "feat_queue",  torch.zeros(self.queue_size, dim,))
        self.register_buffer( "label_queue", torch.full((self.queue_size,), kwargs.get("cls_channels", 21), dtype=torch.int64) )
        self.register_buffer( "iou_queue",   torch.zeros(self.queue_size, ))
        self.register_buffer( "batch_queue", torch.zeros(self.queue_size, dtype=torch.long))  
        self.iou_queue: "torch.Tensor"
        self.label_queue: "torch.Tensor"
        self.feat_queue: "torch.Tensor"
        self.batch_queue : "torch.Tensor"
        # self.register_buffer( "queue_ptr", torch.zeros(1, dtype=torch.long))  # FIFO 指针
        # self.queue_ptr: torch.Tensor
        self.queue_ptr = 0
        self.weight_queue_list = []
        self.usage : int = 0
        self.curr_batch = 0
        if base_feature is not None and osp.exists(base_feature):
            base_feature = torch.load(base_feature)
            merged_records = base_feature['merged_records']
            self.register_buffer("base_feature", F.normalize(torch.as_tensor(merged_records[0]), dim=1))
            self.register_buffer("base_label", torch.as_tensor(merged_records[1]))
            if len(self.base_label.shape) == 1:
                self.base_label = self.base_label.reshape(-1, 1)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.test_id = -1
    def __repr__(self):
        return f"BankSupConLoss (size={self.queue_size}, ptr={self.queue_ptr}, usage={self.usage})"
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, features: "torch.Tensor", labels: "torch.Tensor", ious: "torch.Tensor"):
        self.curr_batch += 1
        batch_size = features.shape[0]
        ptr = self.queue_ptr % self.queue_size
        ## save ptr
        
        self.usage += batch_size
        if self.usage > self.queue_size:
            self.usage = self.queue_size
        batch_1 = self.queue_size - ptr  ### remain size
        
        if batch_size >= batch_1:
            ptr = 0 ### reset and ignore remains batch
            batch_size = min(batch_size, self.queue_size)
            if batch_size < self.queue_size:
                self.feat_queue.copy_(torch.roll(self.feat_queue, shifts=batch_size, dims=0))
                self.label_queue.copy_(torch.roll(self.label_queue, shifts=batch_size))
                self.iou_queue.copy_(torch.roll(self.iou_queue, shifts=batch_size))
                self.batch_queue.copy_(torch.roll(self.batch_queue, shifts=batch_size))
            self.feat_queue [ptr:ptr+batch_size, :] = features
            self.label_queue[ptr:ptr+batch_size] = labels
            self.iou_queue  [ptr:ptr+batch_size] = ious
            self.batch_queue[ptr:ptr+batch_size] = self.curr_batch
            ptr += batch_size
            if ptr >= batch_size: ###
                ptr = 0
        else:
            ## just enqueue
            self.feat_queue [ptr:ptr+batch_size, :] = features
            self.label_queue[ptr:ptr+batch_size] = labels
            self.iou_queue  [ptr:ptr+batch_size] = ious
            self.batch_queue[ptr:ptr+batch_size] = self.curr_batch
            ptr += batch_size
        
        self.queue_ptr = ptr

    def forward(self, features: "torch.Tensor", labels: "torch.Tensor", 
                ious: "torch.Tensor", scores: "torch.Tensor"=None):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        keep = (ious >= self.iou_threshold) 
        labels = labels[keep]
        features = F.normalize(features[keep], dim=1)
        ious = ious[keep]

        if features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)

        ## bsf.b-contrast label
        loss = self._forward(features,   labels,  ious)
        b_loss = self._forward_with_bank(features,   labels,  ious,   )

        if b_loss is not None:
            loss = loss + b_loss

        self._dequeue_and_enqueue(features, labels, ious)
        return loss
        # return torch.tensor(0.0, device=features.device)

    def get_similarity_weight(self, ious, min_weight=0.01):
        weights = torch.full_like(ious, min_weight)
        mask = self.batch_queue > 0
        if not mask.any():
            return weights
        min_batch = self.batch_queue[mask].min().item()
        i = self.curr_batch
        step = 0.01
        init_weight = 1.0
        while i >= min_batch:
            mi = self.batch_queue == i
            weight = init_weight - step * (self.curr_batch - i + 1)
            weights[mi] = max(weight, 0.0)
            i -= 1
        return weights

    def _forward(self, features, labels: "torch.Tensor", ious: "torch.Tensor"):
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.matmul(features, features.T) # T[512, 512]
        similarity = torch.div(similarity, self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0) # 去除自身的相似

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_(1e-5))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        loss = -per_label_log_prob
        coef = self.reweight_func(ious) # * sm
        loss = loss * coef

        return loss.mean()
    
    def _forward_with_bank(self, features: "torch.Tensor", labels: "torch.Tensor", ious):
        
        if self.usage == 0:
            return None
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        with torch.no_grad():
            q_features = self.feat_queue.clone()
            q_labels = self.label_queue.clone()
            q_iou = self.iou_queue.clone()
        if len(q_labels.shape) == 1:
            q_labels = q_labels.reshape(-1, 1)
        label_mask = torch.eq(labels, q_labels.T).float().cuda()
        
        similarity = torch.matmul(features, q_features.T.clone()) # T[894, 8192]
        similarity = torch.div(similarity, self.temperature)
        
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_(1e-5))
        ### similarity weight in memory
        sm = self.get_similarity_weight(q_iou, 0.0)
        
        sl_mask = label_mask.sum(1)
        m_mask = sl_mask == 0
        sl_mask[m_mask] = 1
        per_label_log_prob = (log_prob * sm * label_mask).sum(1) / sl_mask
        
        loss = -per_label_log_prob

        return loss.mean()
    
    def _forward_with_pool(self, features, labels, ious):
        qkeep = torch.randint(0, len(self.base_label), size=(4096, )) 
        qlabels = self.base_label[qkeep]
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        qfeatures = self.base_feature[qkeep]
        features = F.normalize(features, dim=1)
        # qious = self.iou_queue[qkeep]
        label_mask = torch.eq(labels, qlabels.T).float().cuda()
        
        similarity = torch.matmul(features, qfeatures.T) # T[512, 512]
        similarity = torch.div(similarity, self.temperature)
        
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_(1e-5))
        ### similarity weight in memory
        sl_mask = label_mask.sum(1)
        m_mask = sl_mask == 0
        sl_mask[m_mask] = 1
        per_label_log_prob = (log_prob * label_mask).sum(1) / sl_mask
        
        loss = -per_label_log_prob

        return loss.mean()

class SupConLossV2(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, features, labels, ious):
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)


        exp_sim = torch.exp(similarity)
        mask = logits_mask * label_mask
        keep = (mask.sum(1) != 0 ) & (ious >= self.iou_threshold)

        log_prob = torch.log(
            (exp_sim[keep] * mask[keep]).sum(1) / (exp_sim[keep] * logits_mask[keep]).sum(1)
        )

        loss = -log_prob
        return loss.mean()


class SupConLossWithStorage(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, features, labels, ious, queue, queue_label):
        fg = queue_label != -1
        # print('queue', torch.sum(fg))
        queue = queue[fg]
        queue_label = queue_label[fg]

        keep = ious >= self.iou_threshold
        features = features[keep]
        feat_extend = torch.cat([features, queue], dim=0)

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[keep]
        queue_label = queue_label.reshape(-1, 1)
        label_extend = torch.cat([labels, queue_label], dim=0)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, label_extend.T).float().cuda()

        # print('# companies', label_mask.sum(1))

        similarity = torch.div(
            torch.matmul(features, feat_extend.T), self.temperature)
        # print('logits range', similarity.max(), similarity.min())

        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        loss = -per_label_log_prob
        return loss.mean()


class SupConLossWithPrototype(nn.Module):
    '''TODO'''

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, protos, proto_labels):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
            proto (tensor): shape of [B, 128]
            proto_labels (tensor), shape of [B], where B is number of prototype (base) classes
        """
        assert features.shape[0] == labels.shape[0]
        # fg_index = labels != self.num_classes # all feature

        # features = features[fg_index]  # [m, 128]
        # labels = labels[fg_index]      # [m, 128]
        numel = features.shape[0]      # m is named numel

        # m  =  n  +  b
        base_index = torch.eq(labels, proto_labels.reshape(-1,1)).any(axis=0)  # b
        novel_index = ~base_index  # n
        if torch.sum(novel_index) > 1:
            ni_pk = torch.div(torch.matmul(features[novel_index], protos.T), self.temperature)  # [n, B]
            ni_nj = torch.div(torch.matmul(features[novel_index], features[novel_index].T), self.temperature)  # [n, n]
            novel_numer_mask = torch.ones_like(ni_nj)  # mask out self-contrastive
            novel_numer_mask.fill_diagonal_(0)
            exp_ni_nj = torch.exp(ni_nj) * novel_numer_mask  # k != i
            novel_label_mask = torch.eq(labels[novel_index], labels[novel_index].T)
            novel_log_prob = ni_nj - torch.log(exp_ni_nj.sum(dim=1, keepdim=True) + ni_pk.sum(dim=1, keepdim=True))
            loss_novel = -(novel_log_prob * novel_numer_mask * novel_label_mask).sum(1) / (novel_label_mask * novel_numer_mask).sum(1)
            loss_novel = loss_novel.sum()
        else:
            loss_novel = 0

        if torch.any(base_index):
            bi_pi = torch.div(torch.einsum('nc,nc->n', features[base_index], protos[labels[base_index]]), self.temperature) # shape = [b]
            bi_nk = torch.div(torch.matmul(features[base_index], features[novel_index].T), self.temperature)  # [b, n]
            bi_pk = torch.div(torch.matmul(features[base_index], protos.T), self.temperature)  # shape = [b, B]
            # bi_pk_mask = torch.ones_like(bi_pk)
            # bi_pk_mask.scatter_(1, labels[base_index].reshape(-1, 1), 0)
            # base_log_prob = bi_pi - torch.log(torch.exp(bi_nk).sum(1) + (torch.exp(bi_pk) * bi_pk_mask).sum(1))
            base_log_prob = bi_pi - torch.log(torch.exp(bi_nk).sum(1) + torch.exp(bi_pk).sum(1))
            loss_base = -base_log_prob
            loss_base = loss_base.sum()
        else:
            loss_base = 0

        loss = (loss_novel + loss_base) / numel
        try:
            assert loss >= 0
        except:
            print('novel', loss_novel)
            print('base', loss_base)
            exit('loss become negative.')
        return loss

