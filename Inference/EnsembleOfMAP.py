import torch
import numpy as np

from tqdm import trange

class eMAP:
    def __init__(self, objective_fn, param_count, std_init, max_iter, learning_rate, min_lr, device):
        #patience, lr_decay,
        self.objective_fn = objective_fn
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
#        self.patience = patience
#        self.lr_decay = lr_decay
        self.device = device
        self.std_init=std_init
        self.param_count=param_count


    def run(self, ensemble_size):
        theta = torch.Tensor(size=(ensemble_size,self.param_count)).normal_(0.,self.std_init)
        theta=theta.to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
 
        with trange(self.max_iter) as tr:
            for t in tr:
                optimizer.zero_grad()

                loss = -self.objective_fn(theta).sum().squeeze()
                loss.backward()

                lr = optimizer.param_groups[0]['lr']

                tr.set_postfix(loss=loss.item(), lr=lr)

                scheduler.step(loss.detach().clone().cpu().numpy())
                optimizer.step()

                if lr < self.min_lr:
                    break

        return theta.detach()







