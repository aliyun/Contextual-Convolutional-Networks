# -*- coding: utf-8 -*-
'''
Author             : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Date               : 
Last Modified By   : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Last Modified Date : 2023-05-10 17:06
Description        : PyTorch implementation of the loss function for Contextual CNN
-------- 
Copyright (c) 2022 Alibaba Inc. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

_logger = logging.getLogger('train')

from timm.utils import mlog, is_master_rank
from timm.data.mixup import one_hot

avg_ign_loss = False


class TopKBaseLoss(nn.Module):

    def __init__(self,
                 ctx_ks,
                 alphas=1.0,
                 smoothing=[0., 0., 0., 0.],
                 levels=4,
                 ignore_index=-100,
                 use_invariant_scale=False,
                 writer=None,
                 tag='train_or_val'):
        super(TopKBaseLoss, self).__init__()
        self.ctx_ks = ctx_ks
        self.smoothing = smoothing
        if isinstance(alphas, (tuple, list)):
            assert (len(alphas) == levels)
            self.alphas = alphas
        else:
            self.alphas = [alphas] * levels
        self.ignore_index = ignore_index
        self.use_invariant_scale = use_invariant_scale
        self.writer = writer
        self.tag = tag
        assert (len(self.smoothing) == 4 and len(self.ctx_ks) == 4
                and len(self.alphas) == 4)
        mlog(_logger, 'Base-self.alphas: {}'.format(self.alphas))
        mlog(
            _logger, 'Base-self.use_invariant_scale: {}'.format(
                self.use_invariant_scale))

    def get_topk_targets(self, x, target):
        t = target
        targets = [target]  # the first one always uses the raw target. (1x)
        # No need to reindex the target at the final layer (3x)
        recalls = {0: [], 1: [], 2: []}
        num_class = x[0].shape[-1]
        for stage_idx, (p, k) in enumerate(zip(x, self.ctx_ks)):
            if p is not None:
                # Update: for class reranking (instead of class reduction in the paper)
                # get top-k for loss calculation (so that the shape can match with the top-k target)
                ind = torch.topk(p, k=num_class, dim=-1).indices  # (B,k)
                if stage_idx < 3:
                    new_t = []
                    for (_t, _i) in zip(t, ind):
                        if _t in _i:
                            this_ranking = (_i == _t).nonzero(
                                as_tuple=True)[0].squeeze()
                            new_t.append(this_ranking)
                            recalls[stage_idx].append(True)
                        else:
                            new_t.append(
                                torch.ones_like(_t) * self.ignore_index)
                            recalls[stage_idx].append(False)
                    new_t = torch.stack(new_t, dim=0)
                    t = new_t
            else:
                recalls[stage_idx] = None

            if stage_idx < 3:
                targets.append(t)

        return targets

    def get_soft_topk_targets(self, x, target):
        device = target.device
        dtype = target.dtype
        t = target
        targets = [target]
        recalls = {0: [], 1: [], 2: []}
        num_class = x[0].shape[-1]
        for stage_idx, (p, k, s) in enumerate(
                zip(x[:-1], self.ctx_ks[:-1], self.smoothing[:-1])):
            if p is not None:
                # Update: for class reranking (instead of class reduction in the paper)
                # get top-k for loss calculation (so that the shape can match with the top-k target)
                # p = p[:, :k]
                ind = torch.topk(p, k=num_class, dim=-1).indices  # (B,N)
                new_t = []
                for (_t, _i) in zip(t, ind):
                    mix_values, mix_labels = torch.topk(
                        _t, k=2, dim=-1)  # (1,N), two mixed gt labels
                    this_new_t = []

                    ignore_flag = False
                    ignore_mixup_flag = False
                    # check if gt in top-k classes
                    if mix_values.sum() == 0:
                        # this _t is ignored at previous step, continue ignoring
                        ignore_flag = True
                    else:
                        # if miss the main label, ignore; if miss only the mixup label, not ignore``
                        if mix_labels[0] in _i:  # main label
                            this_new_t.append((_i == mix_labels[0]).nonzero(
                                as_tuple=True)[0].squeeze())
                            if mix_labels[1] in _i:  # mixup label
                                this_new_t.append(
                                    (_i == mix_labels[1]).nonzero(
                                        as_tuple=True)[0].squeeze())
                            else:
                                ignore_mixup_flag = True
                        else:
                            ignore_flag = True

                    if ignore_flag:
                        new_t.append(torch.zeros(k, dtype=dtype,
                                                 device=device))
                        recalls[stage_idx].append(False)
                    else:
                        off_value = s / k
                        on_value = 1. - s + off_value
                        if ignore_mixup_flag:
                            lam = 1.
                            y1 = one_hot(this_new_t[0],
                                         k,
                                         on_value=on_value,
                                         off_value=off_value,
                                         device=device)
                            this_new_t = y1.view(-1)
                        else:
                            lam = mix_values[0] / mix_values.sum()
                            y1 = one_hot(this_new_t[0],
                                         k,
                                         on_value=on_value,
                                         off_value=off_value,
                                         device=device)
                            y2 = one_hot(this_new_t[1],
                                         k,
                                         on_value=on_value,
                                         off_value=off_value,
                                         device=device)
                            this_new_t = (y1 * lam + y2 * (1. - lam)).view(-1)
                        new_t.append(this_new_t)
                        recalls[stage_idx].append(True)

                new_t = torch.stack(new_t, dim=0)  #
                t = new_t
            else:
                recalls[stage_idx] = None

            targets.append(t)

        return targets


class TopKCrossEntropy(TopKBaseLoss):
    """ NLL loss with label smoothing.
    """

    def __init__(self, ctx_ks, **kwargs):
        super(TopKCrossEntropy, self).__init__(ctx_ks, **kwargs)
        if any([s > 0. for s in self.smoothing]):
            assert (
                int(torch.__version__.split('.')[0]) >= 1
                and int(torch.__version__.split('.')[1]) >= 10
            ), 'Label smoothing is supported for CE since torch 1.10'  # since 1.10
        self.counter = 0

    def forward(self, x: list, target: torch.Tensor) -> torch.Tensor:
        loss = 0.
        log_loss = 0.
        with torch.no_grad():
            topk_targets = self.get_topk_targets(x, target)
        stage_losses = []
        # mlog(_logger, 'topk_targets', topk_targets)
        assert (len(x) == 4 and len(topk_targets)
                == 4), 'len(x)={}, len(topk_targets)={}'.format(
                    len(x), len(topk_targets))
        prev_k = x[0].shape[-1]  # num_classes
        for i, (p, t, k, a, s) in enumerate(
                zip(x, topk_targets, self.ctx_ks, self.alphas,
                    self.smoothing)):
            if p is None:
                stage_losses.append(None)
                continue
            # Update: for class reranking (instead of class reduction in the paper)
            # get top-k for loss calculation (so that the shape can match with the top-k target)
            # p = p[:, :prev_k]
            # prev_k = k

            l_not_avg_ignore = F.cross_entropy(
                p, t, ignore_index=self.ignore_index, label_smoothing=s) + 1e-6

            if avg_ign_loss:
                l_avg_ignore = F.cross_entropy(p,
                                               t,
                                               ignore_index=self.ignore_index,
                                               label_smoothing=s,
                                               reduction='none').mean() + 1e-6
                l = l_avg_ignore
                log_l = l_not_avg_ignore
            else:
                # or
                l = log_l = l_not_avg_ignore  # early settings

            loss = loss + a * l
            log_loss = log_loss + a * log_l

            stage_losses.append(a * log_l)

        return loss, log_loss, stage_losses


class SoftTargetTopKCrossEntropy(TopKBaseLoss):
    """ NLL loss with label smoothing.
    """

    def __init__(self, ctx_ks, **kwargs):
        super(SoftTargetTopKCrossEntropy, self).__init__(ctx_ks, **kwargs)
        self.counter = 0
        mlog(_logger, 'using SoftTargetTopKCrossEntropy')

    def forward(self, x: list, target: torch.Tensor) -> torch.Tensor:
        loss = 0.
        log_loss = 0.
        with torch.no_grad():
            topk_targets = self.get_soft_topk_targets(x, target)
        stage_losses = []
        assert (len(x) == 4 and len(topk_targets)
                == 4), 'len(x)={}, len(topk_targets)={}'.format(
                    len(x), len(topk_targets))
        prev_k = x[0].shape[-1]  # num_classes
        for i, (p, t, k, a, s) in enumerate(
                zip(x, topk_targets, self.ctx_ks, self.alphas,
                    self.smoothing)):
            if p is None:
                stage_losses.append(None)
                continue

            # Update: for class reranking (instead of class reduction in the paper)
            # get top-k for loss calculation (so that the shape can match with the top-k target)
            p = p[:, :prev_k]
            prev_k = k

            # not avg ignored
            non_ignore_idx = torch.sum(t, dim=-1).nonzero(as_tuple=True)[0]
            count_non_ignore = non_ignore_idx.size(0)
            # replace the mean op with multiplier
            # multiplier -> average only the non-ignored samples, so as to obtain unbias loss.
            if count_non_ignore > 0:
                # (valid_sample,) -> (B,)
                multiplier = F.one_hot(non_ignore_idx,
                                       p.size(0)).sum(0) / count_non_ignore
            else:
                multiplier = 1 / p.size(0)
            l_not_avg_ignore = torch.sum(-t * F.log_softmax(p, dim=-1), dim=-1)
            l_not_avg_ignore = (l_not_avg_ignore * multiplier).sum() + 1e-6

            # avg all
            if avg_ign_loss:
                l_avg_ignore = torch.sum(-t * F.log_softmax(p, dim=-1),
                                         dim=-1).mean() + 1e-6
                l = l_avg_ignore
                log_l = l_not_avg_ignore
            else:
                l = log_l = l_not_avg_ignore  # early settings

            loss = loss + a * l
            log_loss = log_loss + a * log_l

            stage_losses.append(a * log_l)

        return loss, log_loss, stage_losses
