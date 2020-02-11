import torch
import numpy as np


class AdamGradientDescent:
    def __init__(self, objective_fn, max_iter, learning_rate, min_lr, patience, lr_decay, device, verbose):
        self.objective_fn = objective_fn
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device
        self.verbose = verbose

        self._best_theta = None
        self._best_score = None

    def _save_best_model(self, score, theta):
        if score < self._best_score:
            self._best_theta = theta
            self._best_score = score

    def _get_best_model(self):
        return self._best_theta, self._best_score

    def run(self, theta0):
        theta = theta0.detach().clone().to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        self._best_theta = theta.detach().clone().cpu().numpy()
        self._best_score = np.inf
        score = []
        for t in range(self.max_iter - 1):
            optimizer.zero_grad()

            loss = -torch.mean(self.objective_fn(theta))
            loss.backward()

            lr = optimizer.param_groups[0]['lr']

            if self.verbose:
                stats = 'Epoch [{}/{}], Loss: {}, Learning Rate: {}'.format(t, self.max_iter, loss, lr)
                print(stats)

            score.append(loss.detach().clone().cpu().numpy())
            scheduler.step(loss.detach().clone().cpu().numpy())
            optimizer.step()

            self._save_best_model(loss.detach().clone().cpu().numpy(), theta.detach().clone().cpu().numpy())

            if lr < self.min_lr:
                break

        best_theta, best_score = self._get_best_model()
        return best_theta, best_score, score







