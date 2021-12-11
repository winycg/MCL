import torch
from torch import nn
import math


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """

    def __init__(self, args):
        super(ContrastMemory, self).__init__()
        self.number_net = args.number_net
        self.feat_dim = args.feat_dim
        self.n_data = args.n_data
        self.args = args
        self.momentum = args.nce_m
        self.kl = KLDiv(args.kd_T)

        stdv = 1. / math.sqrt(self.feat_dim / 3)
        for i in range(args.number_net):
            self.register_buffer('memory_' + str(i), torch.rand(args.n_data, args.feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, embeddings, pos_idx, neg_idx):
        batchSize = embeddings[0].size(0)
        idx = torch.cat([pos_idx.cuda(), neg_idx.cuda()], dim=1)
        K = self.args.pos_k + self.args.neg_k + 1

        inter_logits = []
        soft_icl_loss = 0.
        for i in range(self.number_net):
            for j in range(i + 1, self.number_net):
                neg_rep = torch.index_select(getattr(self, 'memory_' + str(i)), 0, idx.view(-1)).detach()
                neg_rep = neg_rep.view(batchSize, K, self.feat_dim)
                cos_simi_ij = torch.div(
                    torch.bmm(neg_rep, embeddings[j].view(batchSize, self.feat_dim, 1)).squeeze(-1),
                    self.args.tau)
                inter_logits.append(cos_simi_ij)

                neg_rep = torch.index_select(getattr(self, 'memory_' + str(j)), 0, idx.view(-1)).detach()
                neg_rep = neg_rep.view(batchSize, K, self.feat_dim)
                cos_simi_ji = torch.div(
                    torch.bmm(neg_rep, embeddings[i].view(batchSize, self.feat_dim, 1)).squeeze(-1),
                    self.args.tau)
                inter_logits.append(cos_simi_ji)

                soft_icl_loss += self.kl(cos_simi_ij, cos_simi_ji.detach())
                soft_icl_loss += self.kl(cos_simi_ji, cos_simi_ij.detach())


        mask = torch.zeros(batchSize, 1 + self.args.pos_k + self.args.neg_k).cuda()
        mask[:, :self.args.pos_k+1] = 1.
        icl_loss = 0.
        for logit in inter_logits:
            log_prob = logit - torch.log(torch.exp(logit).sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            icl_loss += - mean_log_prob_pos.mean()


        intra_logits = []
        idx = idx[:, 1:].contiguous()
        K = self.args.pos_k + self.args.neg_k
        for i in range(self.number_net):
            neg_rep = torch.index_select(getattr(self, 'memory_' + str(i)), 0, idx.view(-1)).detach()
            neg_rep = neg_rep.view(batchSize, K, self.feat_dim)
            cos_simi = torch.div(
                torch.bmm(neg_rep, embeddings[i].view(batchSize, self.feat_dim, 1)).squeeze(-1),
                self.args.tau)
            intra_logits.append(cos_simi)

        soft_vcl_loss = 0.
        for i in range(self.number_net):
            for j in range(self.number_net):
                if i != j:
                    soft_vcl_loss += self.kl(intra_logits[i], intra_logits[j].detach())

        vcl_loss = 0.
        mask = torch.zeros(batchSize, self.args.pos_k + self.args.neg_k).cuda()
        mask[:, :self.args.pos_k] = 1.
        for logit in intra_logits:
            log_prob = logit - torch.log(torch.exp(logit).sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            vcl_loss += - mean_log_prob_pos.mean()


        # update memory
        pos_idx = pos_idx[:, 0]
        with torch.no_grad():
            for i in range(len(embeddings)):
                pos = torch.index_select(getattr(self, 'memory_' + str(i)), 0, pos_idx.view(-1))
                pos.mul_(self.momentum)
                pos.add_(torch.mul(embeddings[i], 1 - self.momentum))
                l_norm = pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_v = pos.div(l_norm)
                getattr(self, 'memory_' + str(i)).index_copy_(0, pos_idx, updated_v)

        return vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss


class Sup_MCL_Loss(nn.Module):
    def __init__(self, args):
        super(Sup_MCL_Loss, self).__init__()
        self.embed_list = nn.ModuleList([])
        self.args = args
        if isinstance(args.rep_dim, list):
            for i in range(args.number_net):
                self.embed_list.append(Embed(args.rep_dim[i], args.feat_dim))
        else:
            for i in range(args.number_net):
                self.embed_list.append(Embed(args.rep_dim, args.feat_dim))

        self.contrast = ContrastMemory(args)

    def forward(self, embeddings, pos_idx, neg_idx):

        for i in range(self.args.number_net):
            embeddings[i] = self.embed_list[i](embeddings[i])
        vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss = \
            self.contrast(embeddings, pos_idx, neg_idx)

        return vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


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