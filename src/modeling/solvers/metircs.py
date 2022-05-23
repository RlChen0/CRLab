import numpy as np
import torch
import torchmetrics
from torchmetrics import Metric
from yacs.config import CfgNode
from src.modeling.solvers.metrics.build import METRIC_REGISTRY


@METRIC_REGISTRY.register('Mean')
def build_pearsonr(metric_cfg: CfgNode):
    metric = torchmetrics.MeanMetric()
    return metric


@METRIC_REGISTRY.register('Accuracy')
def build_pearsonr(metric_cfg: CfgNode):
    metric = torchmetrics.Accuracy()
    return metric


@METRIC_REGISTRY.register('R2')
def build_pearsonr(metric_cfg: CfgNode):
    return PearsonR(metric_cfg.NUM_TARGETS, summarize=metric_cfg.SUMMARIZE)


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

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        assert pred.shape == true.shape
        pred = pred.to(torch.float32)
        true = true.to(torch.float32)
        if len(true.shape) == 2:
            reduce_dim = 1
        else:
            reduce_dim = [0, 2]

        product = torch.sum(torch.mul(pred, true), dim=reduce_dim)
        self.product += product

        true_sum = torch.sum(true, dim=reduce_dim)
        self.true_sum += true_sum

        true_sumsq = torch.sum(torch.square(true), dim=reduce_dim)
        self.true_sumsq += true_sumsq

        pred_sum = torch.sum(pred, dim=reduce_dim)
        self.pred_sum += pred_sum

        pred_sumsq = torch.sum(torch.square(pred), dim=reduce_dim)
        self.pred_sumsq += pred_sumsq

        count = torch.ones_like(true)
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


@METRIC_REGISTRY.register('R2')
def build_pearsonr(metric_cfg: CfgNode):
    return R2(metric_cfg.NUM_TARGETS, summarize=metric_cfg.SUMMARIZE)


class R2(Metric):
    def __init__(self, num_targets, dist_sync_on_step=False, summarize=True):
        super(R2, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._summarize = summarize
        self._shape = (num_targets,)
        self.add_state('count', default=torch.zeros(self._shape), dist_reduce_fx=None)

        self.add_state('true_sum', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('true_sumsq', default=torch.zeros(self._shape), dist_reduce_fx=None)

        self.add_state('product', default=torch.zeros(self._shape), dist_reduce_fx=None)
        self.add_state('pred_sumsq', default=torch.zeros(self._shape), dist_reduce_fx=None)

    def update(self, pred: torch.Tensor, true: torch.Tensor, sample_weight=None):
        pred = pred.to(torch.float32)
        true = true.to(torch.float32)
        if len(true.shape) == 2:
            reduce_dim = 1
        else:
            reduce_dim = [0, 2]

        true_sum = torch.sum(true, dim=reduce_dim)
        self.true_sum += true_sum

        true_sumsq = torch.sum(torch.square(true), dim=reduce_dim)
        self.true_sumsq += true_sumsq

        product = torch.sum(torch.mul(true, pred), dim=reduce_dim)
        self.product += product

        pred_sumsq = torch.sum(torch.square(pred), dim=reduce_dim)
        self.pred_sumsq += pred_sumsq

        count = torch.ones_like(true)
        count = torch.sum(count, dim=reduce_dim)
        self.count += count

    def compute(self):
        true_mean = torch.div(self.true_sum, self.count)
        true_mean2 = torch.square(true_mean)

        total = self.true_sumsq - torch.mul(self.count, true_mean2)

        resid1 = self.pred_sumsq
        resid2 = -2 * self.product
        resid3 = self.true_sumsq
        resid = resid1 + resid2 + resid3

        r2 = torch.ones(self._shape, dtype=torch.float32) - torch.div(resid, total)

        if self._summarize:
            return torch.mean(r2)
        else:
            return r2