# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, number_net=2, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.number_net = number_net
        self.ce = nn.CrossEntropyLoss()
        self.kl = KLDiv(T=3)

        # create the encoders
        # num_classes is the output fc dimension
        for i in range(self.number_net):
            setattr(self, 'encoder_q_' + str(i), base_encoder(num_classes=dim))
            setattr(self, 'encoder_k_' + str(i), base_encoder(num_classes=dim))

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q_0.fc.weight.shape[1]
            #for i in range(self.number_net):   
            #self.encoder_q_0.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q_0.fc)
            #self.encoder_q_1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q_1.fc)
            setattr(self, 'encoder_q_' + str(i)+'.fc', nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), getattr(self, 'encoder_q_' + str(i)).fc))
            setattr(self, 'encoder_k_' + str(i)+'.fc', nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), getattr(self, 'encoder_k_' + str(i)).fc))

            #self.encoder_k_0.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k_0.fc)
            #self.encoder_k_1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k_1.fc)

        for i in range(self.number_net):
            for param_q, param_k in zip(getattr(self, 'encoder_q_' + str(i)).parameters(), getattr(self, 'encoder_k_' + str(i)).parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.number_net, dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(self.number_net, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for i in range(self.number_net):
            for param_q, param_k in zip(getattr(self, 'encoder_q_' + str(i)).parameters(), getattr(self, 'encoder_k_' + str(i)).parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, idx):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[idx])
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[idx, :, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[idx] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        queries = []
        for i in range(self.number_net):
            q = getattr(self, 'encoder_q_' + str(i))(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            queries.append(q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            shuffle_keys = []
            for i in range(self.number_net):
                k =  getattr(self, 'encoder_k_' + str(i))(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                shuffle_keys.append(k)

            # undo shuffle
            keys = []
            for key in shuffle_keys:
                k = self._batch_unshuffle_ddp(key, idx_unshuffle)
                keys.append(k)


        
        loss_vcl = 0.
        intra_logits = []
        for i in range(self.number_net):
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [queries[i], keys[i]]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [queries[i], self.queue[i].clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.T
            intra_logits.append(logits)
        
        # labels: positive key indicators
        labels = torch.zeros(intra_logits[0].shape[0], dtype=torch.long).cuda()
        for i in range(self.number_net):
            loss_vcl += self.ce(intra_logits[i], labels)

        loss_soft_vcl = 0.
        for i in range(self.number_net):
            for j in range(self.number_net):
                if i != j:
                    loss_soft_vcl += self.kl(intra_logits[i], intra_logits[j].detach())
        
        loss_icl = 0.
        loss_soft_icl = 0.
        inter_logits = []
        for i in range(self.number_net):
            for j in range(i+1, self.number_net):
                l_pos_ij = torch.einsum('nc,nc->n', [queries[i], keys[j]]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_ij = torch.einsum('nc,ck->nk', [queries[i], self.queue[j].clone().detach()])
                # logits: Nx(1+K)
                logits_ij = torch.cat([l_pos_ij, l_neg_ij], dim=1)
                # apply temperature
                logits_ij /= self.T

                l_pos_ji = torch.einsum('nc,nc->n', [queries[j], keys[i]]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_ji = torch.einsum('nc,ck->nk', [queries[j], self.queue[i].clone().detach()])
                # logits: Nx(1+K)
                logits_ji = torch.cat([l_pos_ji, l_neg_ji], dim=1)
                # apply temperature
                logits_ji /= self.T

                loss_icl += self.ce(logits_ij, labels)
                loss_icl += self.ce(logits_ji, labels)
                loss_soft_icl += self.kl(logits_ij, logits_ji.detach())
                loss_soft_icl += self.kl(logits_ji, logits_ij.detach())
        
        # dequeue and enqueue
        for i in range(len(keys)):
            self._dequeue_and_enqueue(keys[i], i)

        return intra_logits, labels, loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl
        

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
