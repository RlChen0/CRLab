import numpy as np
import torch
from torchmetrics import Metric


class PearsonR(Metric):
    def __init__(self, num_targets, dist_sync_on_step=False, summarize=True):
        super(PearsonR, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._summarize = summarize
        self._shape = (num_targets,)
        self.add_state('count', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('product', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('true_sum', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('true_sumsq', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('pred_sum', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('pred_sumsq', default=torch.zeros(self._shape), dist_reduce_fx=None)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape

        if len(target.shape) == 2:
            reduce_dim = 1
        else:
            reduce_dim = [0, 2]
        product = torch.sum(torch.mul(pred, target), dim=reduce_dim)
        self.product += product

        true_sum = torch.sum(target, dim=reduce_dim)
        self.true_sum += true_sum

        true_sumsq = torch.sum(torch.square(target), dim=reduce_dim)
        self.true_sumsq += true_sumsq

        pred_sum = torch.sum(pred, dim=reduce_dim)
        self.pred_sum += pred_sum

        pred_sumsq = torch.sum(torch.square(pred), dim=reduce_dim)
        self.pred_sumsq += pred_sumsq

        count = torch.ones_like(target)
        count = torch.sum(count, dim=reduce_dim)
        self.count += count

    def compute(self):
        true_mean = torch.div(self.true_sum, self.count)
        true_mean2 = torch.square(true_mean)
        pred_mean = torch.div(self.pred_sum, self.count)
        pred_mean2 = torch.square(pred_mean)

        term1 = self.product
        term2 = -torch.mul(true_mean, self.pred_sum)
        term3 = -torch.mul(pred_mean, self.true_sum)
        term4 = torch.mul(self.count, torch.mul(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = self.true_sumsq - torch.mul(self.count, true_mean2)
        pred_var = self.pred_sumsq - torch.mul(self.count, pred_mean2)
        pred_var = torch.where(
            torch.gt(pred_var, 1e-12),
            pred_var,
            np.inf * torch.ones_like(pred_var)
        )
        tp_var = torch.mul(torch.sqrt(true_var), torch.sqrt(pred_var))
        correlation = torch.div(covariance, tp_var)

        if self._summarize:
            return torch.mean(correlation)
        else:
            return correlation


class R2(Metric):
    def __init__(self,num_targets, dist_sync_on_step=False, summarize=True, )