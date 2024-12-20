import torch
from torch import nn
from torch.nn import functional as F
from criterions.saclr import SACLRBase


class SACLRBatchMix(nn.Module):
    def __init__(self, metric, N, rho, alpha, s_init, single_s, temp):
        super(SACLRBatchMix, self).__init__()
        self.criterion = ExponentialLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s, normalize=True)

    def forward(self, feats, feats_idx):
        return self.criterion(feats, feats_idx)


class ExponentialLoss(SACLRBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp, normalize=True):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
        self.mu = 0.5
        self.normalize = normalize

    def forward(self, feats, feats_idx):
        LARGE_NUM = 1e9
        B = feats.shape[0] // 2
        if self.normalize:
            feats = F.normalize(feats, dim=1, p=2)
            
        feats_a = feats[:B]
        feats_b = feats[B:]

        masks = F.one_hot(torch.arange(B, device=feats_a.device), B)

        logits_aa = -1.0 * torch.cdist(feats_a, feats_a, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = -1.0 * torch.cdist(feats_b, feats_b, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = -1.0 * torch.cdist(feats_a, feats_b, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_ba = -1.0 * torch.cdist(feats_b, feats_a, p=2).pow(2) / (2.0 * self.temp **2.0)

        logits_pos_a = torch.diag(logits_ab)
        logits_pos_b = torch.diag(logits_ba)

        logits_ab = logits_ab - masks * LARGE_NUM
        logits_ba = logits_ba - masks * LARGE_NUM

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        
        q_attr_a = torch.exp(logits_pos_a) #.clamp(min=1e-21) # (B,)
        q_attr_b = torch.exp(logits_pos_b) #.clamp(min=1e-21) # (B,)

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a) #.clamp(min=1e-21) # (B,2B)
        q_rep_b = torch.exp(logits_b) #.clamp(min=1e-21) # (B,2B)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            if self.single_s:
                Zi_a = torch.sum(q_rep_a.detach(), dim=(1,0)) / B
                Zi_b = torch.sum(q_rep_b.detach(), dim=(1,0)) / B
                Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2)
            
            else:
                Zi_a = torch.sum(q_rep_a.detach(), dim=1)
                Zi_b = torch.sum(q_rep_b.detach(), dim=1)
                Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2)

            Z_a = self.mu * Z_hat + (1.0 - self.mu) * Zi_a
            Z_b = self.mu * Z_hat + (1.0 - self.mu) * Zi_b

        repulsive_forces_a = torch.sum(q_rep_a / Z_a.detach().view(-1, 1), dim=1)
        repulsive_forces_b = torch.sum(q_rep_b / Z_b.detach().view(-1, 1), dim=1)

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss

