# -*- coding: utf-8 -*-
'''
Author             : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Date               : 
Last Modified By   : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Last Modified Date : 
Description        : PyTorch implementation of the utility functions for Contextual CNN
-------- 
Copyright (c) 2022 Alibaba Inc. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import reduce

from timm.utils import adaptive_clip_grad


import logging
_logger = logging.getLogger('train')


def is_master_rank():
    try:
        return dist.get_rank()==0
    except:
        return True

def mlog(_logger, *msg):
    if is_master_rank():
        if _logger is None:
            msg = [str(m) for m in msg]
            print(' '.join(msg))
        else:
            msg = [str(m) for m in msg]
            _logger.info(' '.join(msg))


def log_grad(tag, g, writer=None, step=0, dim=(0,)):
    if writer is not None: 
        writer.add_scalar(tag, g.mean(dim=dim).norm(), step)


def get_topk_predictions(x, ctx_ks):
    # get num class
    for _x in x:
        if _x is not None:
            B,N = _x.shape
            pred = torch.zeros_like(_x)
            break
    
    immediate_pred = {}
    labels = torch.arange(N).to(pred.device).unsqueeze(0).expand(B,-1)
    for stage_idx, (p, k) in enumerate(zip(x[:-1], ctx_ks[:-1])): # drop the final ctx_k to keep at least `k` output
        if p is not None:
            if stage_idx<len(x)-1:
                immediate_pred[stage_idx] = torch.scatter(pred, dim=-1, index=labels, src=x[stage_idx])
            # always N since the class reranking update
            ind = torch.topk(p, k=N, dim=-1).indices # (B,N)
            labels = labels.gather(dim=1, index=ind)
            
    pred.scatter_(dim=-1, index=labels, src=x[-1])
    assert(labels.size(-1)>=5), 'To make sure the top-5 acc correctly computed, k of the last ctx should at least 5' 

    return pred, immediate_pred


def get_optim_policies(model, lr, pretrained, weight_decay=0., static_lr_mult=0.1, dynamic_wd_mult=10, filter_bias_and_bn=True, dyn_param_filters=('offset','cls'), enable_emb_policy=False, emb_lr=5e-4, emb_decay=0., downstream_mode=False):
    mlog(_logger, 'pretrained is', pretrained)
    if pretrained or downstream_mode:
        skip = tuple()
        emb_filter = tuple()
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        if hasattr(model, 'emb_filter') and enable_emb_policy:
            emb_filter = model.emb_filter()
        
        static_decay = []
        static_no_decay = []
        dynamic_decay = []
        dynamic_no_decay = []
        emb_policy = []

        mlog(_logger, 'dyn_param_filters is', dyn_param_filters)

        for name, param in model.named_parameters():
            if downstream_mode:
                if not name.startswith('fc'):
                    param.requires_grad = False
                    continue
                
            if not param.requires_grad:
                continue  # frozen weights

            if name in emb_filter:
                mlog(_logger,'No lr decay weight: {}, param shape: {}'.format(name, param.shape))
                emb_policy.append(param)
            elif any([n in name for n in dyn_param_filters]):
                if filter_bias_and_bn:
                    if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
                        dynamic_no_decay.append(param)
                    else:
                        mlog(_logger,'Decayed dynamic weight: {}, param shape: {}'.format(name, param.shape))
                        dynamic_decay.append(param)
                else:
                    dynamic_decay.append(param)
            else:
                if filter_bias_and_bn:
                    if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
                        static_no_decay.append(param)
                    else:
                        static_decay.append(param)
                else:
                    static_decay.append(param)

        param_groups = [{'params': static_decay, 'lr': lr * static_lr_mult, 'weight_decay': weight_decay, 'name': 'static_decay'}]
        if len(static_no_decay)>0:
            param_groups.append({'params': static_no_decay, 'lr': lr * static_lr_mult, 'weight_decay': 0., 'name': 'static_no_decay'})
        if len(dynamic_decay)>0:
            param_groups.append({'params': dynamic_decay, 'lr': lr, 'weight_decay': weight_decay * dynamic_wd_mult, 'name': 'dynamic_decay'})
        if len(dynamic_no_decay)>0:
            param_groups.append({'params': dynamic_no_decay, 'lr': lr, 'weight_decay': 0., 'name': 'dynamic_no_decay'})
        if len(emb_policy)>0:
            param_groups.append({'params': emb_policy, 'lr': emb_lr, 'weight_decay': emb_decay, 'name': 'emb_policy', 'no_lr_decay': True})

        for group in param_groups:                
            mlog(_logger, 'group: {} has {} params, lr: {}, weight decay: {}'.format(
                    group['name'], len(group['params']), group['lr'], group['weight_decay']))
        
        return param_groups

    else:
        return model 
    
    
def config_writer(model, writer, write_freq, args):
    if 'ctx' in args.model:
        is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)
        setattr(model.module if is_ddp else model, 'writer', writer)
        first_ctx_layer = True
        for i, use_ctx in enumerate(args.ctx_layers):
            if use_ctx:
                if first_ctx_layer:
                    first_ctx_layer = False
                else:
                    # for resnet
                    layer = getattr(model.module if is_ddp else model, 'layer%d'%(i+1), None) 
                    if layer is not None:
                        layer.apply(lambda m: setattr(m, 'tag', 'stage{}'.format(i+1)))
                        layer.apply(lambda m: setattr(m, 'writer', None))
                        layer[0].apply(lambda m: setattr(m, 'writer', writer))
                    else:
                        # for convnext
                        layer = model.module.stages[i] if is_ddp else model.stages[i]
                        layer.apply(lambda m: setattr(m, 'tag', 'stage{}'.format(i+1)))
                        layer.apply(lambda m: setattr(m, 'writer', None))
                        # layer.blocks[0].apply(lambda m: setattr(m, 'writer', writer)) # timm
                        layer[0].apply(lambda m: setattr(m, 'writer', writer)) # official
        model.apply(lambda m: setattr(m, 'write_freq', write_freq)) 
