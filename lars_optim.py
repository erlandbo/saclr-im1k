# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# https://github.com/facebookresearch/barlowtwins/blob/main/main.py
# https://github.com/Optimization-AI/SogCLR/blob/PyTorch/sogclr/optimizer.py


import torch
from torch.optim import Optimizer

class LARS(Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])
                
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

