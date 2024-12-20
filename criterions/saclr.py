import torch
from torch import nn
from torch.nn import functional as F


class SACLR(nn.Module):
    def __init__(self, metric, N, rho, alpha, s_init, single_s, temp):
        super(SACLR, self).__init__()
        self.criterion = ExponentialLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s, normalize=True)

    def forward(self, feats, feats_idx):
        return self.criterion(feats, feats_idx)


class SACLRBase(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLRBase, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.temp = temp
        self.single_s = single_s

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a  
        E_attr_b = q_attr_b  

        E_rep_a = torch.sum(q_rep_a, dim=1) / (2.0 * B - 2.0)  
        E_rep_b = torch.sum(q_rep_b, dim=1) / (2.0 * B - 2.0)  
        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B  

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a 
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0



class ExponentialLoss(SACLRBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp, normalize=True):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
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
        
        q_attr_a = torch.exp(logits_pos_a)  # .clamp(min=1e-21) # (B,)
        q_attr_b = torch.exp(logits_pos_b)  #.clamp(min=1e-21) # (B,)

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a)  #.clamp(min=1e-21) # (B,2B)
        q_rep_b = torch.exp(logits_b)  #.clamp(min=1e-21) # (B,2B)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2) 

        repulsive_forces_a = torch.sum(q_rep_a / Z_hat.detach().view(-1,1), dim=1) 
        repulsive_forces_b = torch.sum(q_rep_b / Z_hat.detach().view(-1,1), dim=1) 

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss


