import torch
from torch import nn
from torch.nn import functional as F


class SCL(nn.Module):
    def __init__(self, metric, N, rho, alpha, s_init, single_s, temp):
        super(SCL, self).__init__()
        if metric == "exponential":
            self.criterion = ExponentialLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s, normalize=True)
        elif metric == "cauchy":
            self.criterion = CauchyLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s)
        else:
            raise ValueError(f"Invalid metric: {metric}")

    def forward(self, feats, feats_idx):
        return self.criterion(feats, feats_idx)


class SCLBase(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SCLBase, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.single_s = single_s
        self.temp = temp

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a 
        E_attr_b = q_attr_b 

        E_rep_a = q_rep_a 
        E_rep_b = q_rep_b 

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


class ExponentialLoss(SCLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp, normalize=True):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
        self.normalize = normalize

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1) 
            
        feats_a = feats[:B]
        feats_b = feats[B:]
        
        q_attr_a = torch.exp( -1.0 * F.pairwise_distance(feats_a, feats_b, p=2).pow(2) / (2.0 * self.temp**2.0) )  
        q_attr_b = torch.exp( -1.0 * F.pairwise_distance(feats_b, feats_a, p=2).pow(2) / (2.0 * self.temp**2.0) )  
        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)
        
        neg_idxs = torch.roll(torch.arange(B), shifts=-1, dims=0)
        q_rep_a = torch.exp( -1.0 * F.pairwise_distance(feats_a, feats_b[neg_idxs], p=2).pow(2) / (2.0 * self.temp**2.0) )  
        q_rep_b = torch.exp( -1.0 * F.pairwise_distance(feats_b, feats_a[neg_idxs], p=2).pow(2) / (2.0 * self.temp**2.0) )  

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Z_hat = self.s_inv[feats_idx] / self.N.pow(2)

        repulsive_forces_a = q_rep_a / Z_hat
        repulsive_forces_b = q_rep_b / Z_hat

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss



class CauchyLoss(SCLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        feats_a = feats[:B]
        feats_b = feats[B:]
        
        q_attr_a = 1.0 / (1.0 + F.pairwise_distance(feats_a, feats_b, p=2).pow(2)) 
        q_attr_b = 1.0 / (1.0 + F.pairwise_distance(feats_b, feats_a, p=2).pow(2))  
        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)
        
        neg_idxs = torch.roll(torch.arange(B), shifts=-1, dims=0)
        q_rep_a = 1.0 / (1.0 + F.pairwise_distance(feats_a, feats_b[neg_idxs], p=2).pow(2)) 
        q_rep_b = 1.0 / (1.0 + F.pairwise_distance(feats_b, feats_a[neg_idxs], p=2).pow(2))  

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Z_hat = self.s_inv[feats_idx] / self.N.pow(2) 

        repulsive_forces_a = q_rep_a / Z_hat
        repulsive_forces_b = q_rep_b / Z_hat

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss