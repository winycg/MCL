import torch
from torch import nn
import math
import torch.nn.functional as F


class SupMCL(nn.Module):
    def __init__(self, args):
        super(SupMCL, self).__init__()
        self.number_net = args.number_net
        self.feat_dim = args.feat_dim
        self.args = args
        self.kl = KLDiv(T=args.kd_T)

    def forward(self, embeddings, labels):
        batchSize = embeddings[0].size(0)

        labels = labels.unsqueeze(0)
        intra_mask = torch.eq(labels, labels.T).float() - torch.eye(labels.size(1)).cuda()
        inter_mask = torch.eq(labels, labels.T).float()
        diag_mask = (1.-torch.eye(labels.size(1)).cuda())

        inter_logits = []
        soft_icl_loss = 0.
        for i in range(self.number_net):
            for j in range(i + 1, self.number_net):
                cos_simi_ij = torch.div(
                    torch.mm(embeddings[i], embeddings[j].T),
                    self.args.tau)
                inter_logits.append(cos_simi_ij)

                cos_simi_ji = torch.div(
                    torch.mm(embeddings[j], embeddings[i].T),
                    self.args.tau)
                inter_logits.append(cos_simi_ji)

                soft_icl_loss += self.kl(cos_simi_ij, cos_simi_ji.detach())
                soft_icl_loss += self.kl(cos_simi_ji, cos_simi_ij.detach())

        icl_loss = 0.
        for logit in inter_logits:
            log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            icl_loss += - mean_log_prob_pos.mean()

        
        intra_logits = []
        for i in range(self.number_net):
            cos_simi = torch.div(
                torch.mm(embeddings[i], embeddings[i].T),
                self.args.tau)
            intra_logits.append(cos_simi)

        soft_vcl_loss = 0.
        for i in range(self.number_net):
            for j in range(self.number_net):
                if i != j:
                    soft_vcl_loss += self.kl(intra_logits[i], intra_logits[j].detach())

        vcl_loss = 0.
        for logit in intra_logits:
            log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            vcl_loss += - mean_log_prob_pos.mean()

        return vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss


class Sup_MCL_Loss(nn.Module):
    def __init__(self, args):
        super(Sup_MCL_Loss, self).__init__()
        self.embed_list = nn.ModuleList([])
        self.args = args
        for i in range(args.number_net):
            self.embed_list.append(Embed(args.rep_dim[i], args.feat_dim))

        self.contrast = SupMCL(args)

    def forward(self, embeddings, labels):

        for i in range(self.args.number_net):
            embeddings[i] = self.embed_list[i](embeddings[i])
        vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss = \
            self.contrast(embeddings, labels)

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
