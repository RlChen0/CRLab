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
        pred, target = self._input_format(pred, target)
        assert pred.shape == target.shape

        if len(target.shape) == 2:
            reduce_dim = 0
        else:
            reduce_dim = [0, 2]
        product = torch.sum(torch.dot(pred, target), dim=reduce_dim)
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
        term2 = -torch.dot(true_mean, self.pred_sum)
        term3 = -torch.dot(pred_mean, self.true_sum)
        term4 = torch.dot(self.count, torch.dot(true_mean, pred_mean))

        true_var = self.true_sumsq - torch.dot(self.count, true_mean2)
        pred_var = self.pred_sumsq - torch.dot(self.count, pred_mean2)
        pred_var = torch.where




