# -*- coding: utf-8 -*-
'''
Author             : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Date               : 
Last Modified By   : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Last Modified Date : 2023-05-12 16:40
Description        : PyTorch implementation of Contextual ResNet
                     (modified from https://github.com/huggingface/pytorch-image-models)
-------- 
Copyright (c) 2022 Alibaba Inc. 
'''

import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier, create_pool
from .layers import ClassProjectHead, NonlinearMeanPool1d, VisualProjectHead, StaticConv2d, WeightOffsetConv2d, SpatialOffsetConv2d, ContextualConv2d
from .registry import register_model
from .resnet import Bottleneck, drop_blocks, downsample_conv, create_aa, downsample_avg
from functools import reduce

import logging
_logger = logging.getLogger('train')
from timm.utils import mlog, log_grad, get_topk_predictions


    

def count_params(model):
    return sum([m.numel() for m in model.parameters()])

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'cls', 
        'strict_load': False,
        **kwargs
    }

default_cfgs = {
    'ctx_r18': _cfg(url='', interpolation='bicubic', crop_pct=0.95),
    'ctx_r34': _cfg(url='', interpolation='bicubic', crop_pct=0.95),
    'ctx_r50': _cfg(url='', interpolation='bicubic', crop_pct=0.95),
}


class DynamicBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 conv_fn, 
                 class_emb_dim, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 cardinality=1, 
                 base_width=64,
                 reduce_first=1, 
                 dilation=1, 
                 first_dilation=None, 
                 act_layer=nn.ReLU, 
                 norm_layer=nn.BatchNorm2d,
                 attn_layer=None, 
                 aa_layer=None, 
                 drop_block=None, 
                 drop_path=None,
                 include_context_for_offset=True,
                 use_mask=True,
                 writer=None):
        super(DynamicBasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = conv_fn(
            class_emb_dim=class_emb_dim, in_channels=inplanes, out_channels=first_planes, 
            kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, 
            bias=False, 
            norm_fn=(nn.BatchNorm1d, nn.BatchNorm2d),
            include_context_for_offset=include_context_for_offset, use_mask=use_mask, writer=writer)

        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            class_emb_dim=class_emb_dim, 
            in_channels=first_planes, 
            out_channels=outplanes, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation, 
            bias=False, 
            norm_fn=(nn.BatchNorm1d, nn.BatchNorm2d),
            include_context_for_offset=include_context_for_offset, use_mask=use_mask, writer=writer)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, inputs):
        x, context = inputs
        shortcut = x

        x = self.conv1(x, context)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return (x, context) # keep the input to next module of nn.Sequential consistent.
        
class DynamicBottleneck(nn.Module):
    """DynamicBottleneck for Contextual ResNet (create a dynamic attn weights)"""
    expansion = 4
    def __init__(self, 
                 conv_fn, 
                 class_emb_dim, 
                 inplanes, 
                 planes,
                 stride=1, 
                 downsample=None, 
                 cardinality=1, 
                 base_width=64,
                 reduce_first=1, 
                 dilation=1, 
                 first_dilation=None, 
                 act_layer=nn.ReLU, 
                 norm_layer=nn.BatchNorm2d, 
                 attn_layer=None, 
                 aa_layer=None, 
                 drop_block=None, 
                 drop_path=None,
                 include_context_for_offset=True,
                 use_mask=True,
                 writer=None):
        super(DynamicBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = conv_fn(
            class_emb_dim=class_emb_dim, 
            in_channels=first_planes, 
            out_channels=width, 
            kernel_size=3, 
            stride=1 if use_aa else stride,
            padding=first_dilation, 
            dilation=first_dilation, 
            groups=cardinality, 
            bias=False, 
            norm_fn=(nn.BatchNorm1d, nn.BatchNorm2d),
            include_context_for_offset=include_context_for_offset, 
            use_mask=use_mask, 
            writer=writer)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, inputs):
        x, context = inputs
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x, context)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return (x, context) # keep the input to next module of nn.Sequential consistent.

        
def make_blocks(
        dyn_block_fn, dyn_conv_fn, class_emb_dim, channels, block_repeats, inplanes, 
        dyn_layers=(1,1,1,1), include_context_for_offset=False, use_mask=True, dyn_block_freq=1, 
        reduce_first=1, output_stride=32, down_kernel_size=1, avg_down=False, drop_block_rate=0., 
        drop_path_rate=0., writer=None, **kwargs):
    """make contextual residual blocks for each stage of the backbone
    """    
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    static_conv_fn = StaticConv2d

    first_ctx_layer = False if class_emb_dim is None else True # class_emb_dim is None denotes resnet_dyn_conv 
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        # Note: resnet_dyn_conv with dyn_layers=[0,0,0,1] ~= Contextual ResNet with dyn_layers=[0,0,1,1]
        if dyn_layers[stage_idx]:
            if first_ctx_layer:
                conv_fn = static_conv_fn
                first_ctx_layer = False
            else:
                conv_fn = dyn_conv_fn
        else:
            conv_fn = static_conv_fn
        
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        # for res2-res5, use downsample_conv at end of the first block, 
        # which is 1x1conv (w/ stride=2) + norm
        downsample = None
        if stride != 1 or inplanes != planes * dyn_block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * dyn_block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs) 
            

        block_kwargs = dict(include_context_for_offset=include_context_for_offset, use_mask=use_mask, reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):

            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(dyn_block_fn(
                conv_fn if block_idx%dyn_block_freq==0 else static_conv_fn, 
                class_emb_dim, 
                inplanes, 
                planes, 
                stride, 
                downsample, 
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, writer=writer, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * dyn_block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info



class Contextualizing(nn.Module):
    """The contextualizing layer for Contextual CNN"""

    def __init__(self, 
                num_features, 
                num_classes, 
                ctx_layers, 
                ctx_ks, 
                ctx_rank_loss, 
                dyn_conv_fn, 
                class_merge_fn, 
                pool_type='avg', 
                use_conv=False, 
                input_type='label_index', 
                drop_rate=0.0, 
                act_layer=nn.ReLU, 
                norm_layer=nn.BatchNorm2d, 
                class_emb_dim=512, 
                use_class_proj=False,
                lv_cls_emb=False,
                logit_scale_method='fixed',
                emb_init_method='clip',
                include_context_for_offset=True,
                input_resolution=224, 
                downsample=True,
                writer=None,
                block_expansion=1,
                cls_norm_fn=nn.BatchNorm1d):
        self.input_type = input_type
        self.ctx_layers = ctx_layers
        self.ctx_ks = ctx_ks
        self.drop_rate = drop_rate if isinstance(drop_rate, tuple) else [drop_rate] * len(ctx_ks)
        self.emb_init_method = emb_init_method
        self.writer = writer
        self.num_classes = num_classes
        self.lv_cls_emb = lv_cls_emb
        super(Contextualizing, self).__init__()

        if input_type is 'label_index': # learnable randomly-initialized parameters
            # self.class_token = torch.nn.Parameter(torch.arange(num_classes), requires_grad=False)
            self.class_token = torch.arange(num_classes) # not parameter
            if not self.lv_cls_emb:
                self.class_token_embedding = nn.Embedding(num_classes, class_emb_dim) # (#classes, C)
                self.mapper = None
            else:
                self.mapper = {}
                count = 0
                for i, use_ctx in enumerate(ctx_layers):
                    if use_ctx:
                        self.mapper[i] = (count*class_emb_dim, (count+1)*class_emb_dim)
                        count+=1
                    else:
                        self.mapper[i] = None
                total_class_dim = sum(ctx_layers) * class_emb_dim
                self.class_token_embedding = nn.Embedding(num_classes, total_class_dim) 
            mlog(_logger, 'self.class_token_embedding', self.class_token_embedding)
                
        elif input_type is 'label_text':
            raise NotImplementedError

        self.class_encoder = nn.Identity()
        self.visual_proj_heads = nn.ModuleList()
        self.class_mergers = nn.ModuleList() 
        self.class_proj_heads = nn.ModuleList()

        
        class_proj_head = ClassProjectHead
        vis_head = VisualProjectHead
        if class_merge_fn == 'nonlinear_mean':
            class_merge_head = NonlinearMeanPool1d
        else:
            raise NotImplementedError
        
        mlog(_logger, 'vis_head=', vis_head, 'class_merge_head=', class_merge_head, 'class_proj_head=',class_proj_head)

        is_static_backbone = dyn_conv_fn==StaticConv2d or include_context_for_offset == False # all vanilla convs or not include context for offsets.
        prev_k = num_classes
        first_ctx_layer = True
        
        for stage_idx, (use_ctx, k) in enumerate(zip(ctx_layers, ctx_ks)):
            if use_ctx:
                # Follow CLIP.
                if isinstance(num_features, (list, tuple)):
                    this_feature_dim = num_features[stage_idx]
                else:
                    this_feature_dim = num_features // (2**(len(ctx_layers)-stage_idx-1))
                this_resolution = input_resolution // (2**(stage_idx+2)) 

                if use_class_proj: # default False
                    self.class_proj_heads.append(
                        class_proj_head(
                            embed_dim=class_emb_dim,
                            hidden_dim=this_feature_dim//block_expansion,
                            output_dim=class_emb_dim
                        )
                    )
                else:
                    self.class_proj_heads.append(
                            nn.Identity()
                    )

                self.visual_proj_heads.append(
                    vis_head(spacial_dim=this_resolution, 
                        embed_dim=this_feature_dim, 
                        hidden_dim=this_feature_dim//block_expansion,
                        output_dim=class_emb_dim,
                        norm_fn=cls_norm_fn
                        )
                )
                
                if is_static_backbone==False and first_ctx_layer==False:
                    this_out_dim = this_feature_dim//(block_expansion*2)
                    self.class_mergers.append(class_merge_head(
                        num_classes=prev_k, 
                        embed_dim=class_emb_dim, 
                        output_dim=this_out_dim,
                        merge_norm_fn=cls_norm_fn)) # fixme: cls_norm_fn
                else:
                    # The first ctx layer (1000 input classes) do not use context as well, since the input classes are the same for all samples)
                    self.class_mergers.append(nn.Identity()) 
                prev_k = k if k != -1 else prev_k
                first_ctx_layer = False
            else:
                self.visual_proj_heads.append(nn.Identity())
                self.class_mergers.append(nn.Identity())
                self.class_proj_heads.append(nn.Identity())

        self.ctx_rank_loss = ctx_rank_loss
        if self.ctx_rank_loss is False:
            if logit_scale_method == 'fixed':
                self.logit_scale = 1 / 0.07
            elif logit_scale_method == 'share':
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            elif logit_scale_method == 'stagewise':
                self.logit_scale = nn.ParameterList([nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=bool(use_ctx)) for use_ctx in ctx_layers])
            elif logit_scale_method == 'recall-precision':
                self.logit_scale = nn.ParameterList([nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) for _ in range(2)])
        else:
            self.logit_scale = None

        self.init_weights()
        self.features = None
        # self.features = {'stage1':[], 'stage2':[], 'stage3':[], 'stage4':[], 'labels':[]}

    def init_weights(self):
        if self.emb_init_method == 'clip':
            nn.init.normal_(self.class_token_embedding.weight, std=0.02)
            mlog(_logger, 'Initialize class embeddings following CLIP')
        elif self.emb_init_method == 'linear':
            nn.init.kaiming_uniform_(self.class_token_embedding.weight, a=math.sqrt(5))
            mlog(_logger, 'Initialize class embeddings following FC')
        else:
            mlog(_logger, 'Initialize class embeddings following default behavior of nn.Embedding layer')



    def generate(self, device):
        c = self.class_token_embedding(self.class_token.to(device))
        return c

    def merge(self, c, stage_idx, k):
        # Update: Class reranking (compared to class reduction in the paper) 
        # leads to more stable convergence.
        this_c = c[:,:k,:]
        return self.class_mergers[stage_idx](this_c)

    def classify(self, x, c, stage_idx):   
        assert(len(x.size())==4 and len(c.size())==3), 'x.shape:{}, c.shape:{}'.format(x.shape, c.shape)
        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.exp()  
        elif isinstance(self.logit_scale, nn.ParameterList):
            if len(self.logit_scale)==4:
                logit_scale = self.logit_scale[stage_idx].exp()  
            elif len(self.logit_scale)==2:
                logit_scale = self.logit_scale[(stage_idx+1)//4].exp()  
            else:
                raise NotImplementedError
        else:
            logit_scale = self.logit_scale or 1.


        vis_proj_head = self.visual_proj_heads[stage_idx] 
        class_proj_head = self.class_proj_heads[stage_idx] 
            
        x = vis_proj_head(x) # (B,C,h,w) -> (B,C), proj to class space
        x = x.unsqueeze(1) # (B,C) -> (B,1,C)
        norm_x = x / x.norm(dim=-1, keepdim=True) # (B,1,C)
        
        # for vis
        if not self.training and self.features is not None:
            self.features['stage{}'.format(stage_idx+1)] += [f for f in norm_x.squeeze().cpu().numpy()]

        proj_c = class_proj_head(c)
        norm_c = proj_c / proj_c.norm(dim=-1, keepdim=True) # (B,N,C)

        # cosine similarity as logits
        y = logit_scale * (norm_x @ norm_c.transpose(1,2)) # (B,1,C)@(B,C,N)->(B,1,N)        
        y = y.squeeze(1) # (B,1,N)->(B,N)

        # Update: Applying class reranking (compared to class reduction in the paper) 
        # leads to more stable convergence.
        idx = torch.topk(y, self.num_classes, dim=-1).indices.unsqueeze(-1).expand(-1,-1,c.shape[-1])
        c = c.gather(dim=1, index=idx)

        return y, c

    def forward(self, x, c, stage_idx):
        return self.classify(x, c, stage_idx)


class Contextual_ResNet(nn.Module):
    """Contextual_ResNet
    
    This class implements all variants of Contextual_ResNet

    Contextual_ResNet variants:
     * 
     * 

    Parameters
    ----------
    block : Block
        Class for the basic block (res1/res2). Options are .
    dyn_block : Block
        Class for the ctx block (res3/res4/res5). Options are .
    layers : list of int
        Numbers of layers in each block
    """

    def __init__(self, 
                 dyn_block, 
                 dyn_conv_fn, 
                 layers, 
                 num_classes=1000,
                 in_chans=3,
                 ctx_layers=(1, 1, 1, 1), 
                 ctx_ks=(500,200,50,1), 
                 dyn_block_freq=1,
                 class_emb_dim=256, 
                 class_merge_fn='nonlinear_mean',
                 include_context_for_offset=True,
                 use_mask=True,
                 use_class_proj=False,
                 lv_cls_emb=False,
                 logit_scale_method='fixed',
                 emb_init_method='clip',
                 ctx_rank_loss=False, 
                 cardinality=1, 
                 base_width=64, 
                 stem_width=64, 
                 stem_type='', 
                 replace_stem_pool=False,
                 output_stride=32, 
                 block_reduce_first=1, 
                 down_kernel_size=1, 
                 avg_down=False,
                 act_layer=nn.ReLU, 
                 norm_layer=nn.BatchNorm2d, 
                 aa_layer=None, 
                 drop_rate=0.0, 
                 drop_path_rate=0.,
                 drop_block_rate=0., 
                 global_pool='avg', 
                 zero_init_last_bn=True, 
                 strict_load=False,
                 block_args=None, 
                 cls_args=None,
                 drop_without_ctx=False, # not used for resnet
                 ### linear probe
                 downstream_mode=False, 
                 downstream_with_proj=False,
                 downstream_with_multistage=False,
                 downstream_num_multistage=4,
                 downstream_num_classes=200, 
                 writer=None):
        block_args = block_args or dict()
        self.cls_args = cls_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.ctx_layers = ctx_layers
        self.ctx_ks = tuple(ctx_ks)   
        self.dyn_conv_fn = dyn_conv_fn
        self.class_merge_fn = class_merge_fn
        self.strict_load = strict_load
        self.ctx_rank_loss = ctx_rank_loss
        self.downstream_with_proj = downstream_with_proj
        self.downstream_with_multistage = downstream_with_multistage
        # self.writer = writer
        self.writer = None
        super(Contextual_ResNet, self).__init__()
        mlog(_logger, 'self.ctx_layers: {}'.format(self.ctx_layers))

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2),
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            dyn_block, dyn_conv_fn, class_emb_dim, channels, layers, inplanes, 
            dyn_layers=self.ctx_layers, include_context_for_offset=include_context_for_offset, use_mask=use_mask, 
            dyn_block_freq=dyn_block_freq,
            cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, writer=self.writer, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)
        

        # Head (Pooling and Classifier)
        self.num_features = 512 * dyn_block.expansion
        self.cls_args['class_emb_dim'] = class_emb_dim
        self.cls_args['use_class_proj'] = use_class_proj
        self.cls_args['lv_cls_emb'] = lv_cls_emb
        self.cls_args['include_context_for_offset'] = include_context_for_offset
        self.cls_args['logit_scale_method'] = logit_scale_method
        self.cls_args['emb_init_method'] = emb_init_method
        self.cls_args['block_expansion'] = dyn_block.expansion
        self.cls_args['writer'] = writer
        self.cls = Contextualizing(
            self.num_features, 
            self.num_classes, 
            self.ctx_layers, 
            self.ctx_ks, 
            self.ctx_rank_loss, 
            self.dyn_conv_fn, 
            self.class_merge_fn, 
            **self.cls_args)
        
        if downstream_mode:
            self.downstream_num_multistage = downstream_num_multistage
            fc_in_dim = self.num_features
            if self.downstream_with_proj:
                fc_in_dim = class_emb_dim
            if self.downstream_with_multistage:
                if self.downstream_num_multistage==4:
                    fc_in_dim = (self.num_features * 15)//8
                elif self.downstream_num_multistage==3:
                    fc_in_dim = (self.num_features * 14)//8
                elif self.downstream_num_multistage==2:
                    fc_in_dim = (self.num_features * 12)//8
            
            self.fc = nn.Linear(fc_in_dim, downstream_num_classes, bias=True)
            mlog(_logger, 'Downstream Mode=True! Enable self.fc:', self.fc)

        mlog(_logger, 'Contextual_ResNet:', self)


        self.init_weights(zero_init_last_bn=zero_init_last_bn)
        self.visualization = False
        self.single_output = False
        self.intermediate_featrues = []
                
    def no_weight_decay(self):
        return ('cls.class_token', 'cls.logit_scale', 'cls.logit_scale.0', 'cls.logit_scale.1', 'cls.logit_scale.2', 'cls.logit_scale.3', 'cls.class_token_embedding.weight',)

    def emb_filter(self):
        return ('cls.class_token_embedding.weight',)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.
        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        init_state_dict = self.state_dict()
        replacements = []
        adds = []
        mismatches = []
        strict = strict and self.strict_load
        static_to_dynamic_rules = {
            'conv1.weight': 'conv1.conv.weight',
            'conv2.weight': 'conv2.conv.weight',
            'conv3.weight': 'conv3.conv.weight',
        }
        for k, v in init_state_dict.items():
            if k not in state_dict.keys():
                # check whether the corresponding static weights exists
                not_find_replacement_flag = True
                for old, new in static_to_dynamic_rules.items():
                    static_k = k.replace(new, old)
                    if static_k in state_dict.keys():
                        if state_dict[static_k].shape == init_state_dict[k].shape:
                            state_dict[k] = state_dict[static_k] # rename static weight with new name
                            replacements.append(k)
                            del state_dict[static_k]
                        else:
                            if not strict: # which won't throw exception
                                mismatches.append(k)
                                state_dict[k] = init_state_dict[k] # use random init weights
                        not_find_replacement_flag = False
                        break

                # not find any replacement, then denoted by new weights
                if not_find_replacement_flag:
                    adds.append(k)
                    state_dict[k] = init_state_dict[k]  # use random init weights

            else:
                if not strict: # which won't throw exception        
                    if state_dict[k].shape != init_state_dict[k].shape:
                        mismatches.append(k)
                        state_dict[k] = init_state_dict[k] # use random init weights

        
        mlog(_logger, 'Find Replacements Weights: {}'.format(replacements))
        mlog(_logger, 'Newly Added Weights: {}'.format(adds))
        mlog(_logger, 'Mismatched Weights (Used only when not strict): {}'.format(mismatches))     

        # Below is the default loading               
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

    def init_weights(self, zero_init_last_bn=True):
        for n, m in self.named_modules():
            if 'offset' in n:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()
        
    def get_classifier(self):
        return self.cls

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.cls = Contextualizing(
            self.num_features, 
            self.num_classes, 
            self.ctx_layers, 
            self.ctx_ks, 
            self.ctx_rank_loss, 
            self.dyn_conv_fn, 
            self.class_merge_fn, 
            **self.cls_args)


    def forward_features(self, x):
        B = x.size(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        c = self.cls.generate(device=x.device).unsqueeze(0).expand(B,-1,-1)

        flag1, flag2, flag3, flag4 = self.ctx_layers
        n2, n3, n4, _ = self.ctx_ks
        if self.downstream_with_multistage:
            xs = []
        ctx = None
        x, _ = self.layer1((x, ctx))
        if self.downstream_with_multistage and self.downstream_num_multistage>=4: 
            xs.append(x.mean([-2,-1], keepdim=True))
        if flag1:
            _, c = self.cls(x, c, k1, stage_idx=0)
            ctx = self.cls.merge(c, 1, n2) 
        x, _ = self.layer2((x, ctx))
        if self.downstream_with_multistage and self.downstream_num_multistage>=3:  
            xs.append(x.mean([-2,-1], keepdim=True))

        if flag2:
            _, c = self.cls(x, c, k2, stage_idx=1)
            ctx = self.cls.merge(c, 2, n3)
        x, _ = self.layer3((x, ctx))
        if self.downstream_with_multistage and self.downstream_num_multistage>=2: 
            xs.append(x.mean([-2,-1], keepdim=True))

        if flag3:
            _, c = self.cls(x, c, k3, stage_idx=2)
            ctx = self.cls.merge(c, 3, n4)
        x, _ = self.layer4((x, ctx))
        if self.downstream_with_multistage: 
            xs.append(x.mean([-2,-1], keepdim=True))
            x = torch.cat(xs, dim=1)
            
        return x

    def forward_downstream(self, x):
        with torch.no_grad():
            x = self.forward_features(x)
            if self.downstream_with_proj:
                x = self.cls.visual_proj_heads[3](x)
                norm_x = x / x.norm(dim=-1, keepdim=True)
            else:
                x = x.mean([-2,-1])
        x = self.fc(x)
        return x

    def forward(self, x):
        B = x.size(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        c = self.cls.generate(device=x.device).unsqueeze(0).expand(B,-1,-1)
        y = [None,] * 4

        flag1, flag2, flag3, flag4 = self.ctx_layers
        n2, n3, n4, _ = self.ctx_ks

        ctx = None
        x, _ = self.layer1((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))
        
        if flag1:
            y[0], c = self.cls(x, c, stage_idx=0)
            ctx = self.cls.merge(c, 1, n2) 
        x, _ = self.layer2((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        if flag2:
            y[1], c = self.cls(x, c, stage_idx=1)
            ctx = self.cls.merge(c, 2, n3)
        x, _ = self.layer3((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        if flag3:
            y[2], c = self.cls(x, c, stage_idx=2)
            ctx = self.cls.merge(c, 3, n4)
        x, _ = self.layer4((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        y[3], c = self.cls(x, c, stage_idx=3)
            
        if self.visualization and self.single_output:
            vis_y, _ = get_topk_predictions(y, self.ctx_ks)
            return vis_y
        return y


def _create_ctx_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        Contextual_ResNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)

CONV_FNS_MAP = {
    1: None,
    2: None,
    3: WeightOffsetConv2d,
    4: SpatialOffsetConv2d,
    5: ContextualConv2d
}

model_urls = {
    "ctx_r18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "ctx_r34": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth",
    "ctx_r50": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",

}

@register_model
def ctx_r50(pretrained=False, **kwargs):
    """Constructs a Contextual ResNet-50 model.
    """
    
    # Configure conv functions in DynamicBottleneck
    dyn_conv_fn = CONV_FNS_MAP[kwargs['conv_fn']]
    del kwargs['conv_fn']
        
    model_args = dict(dyn_block=DynamicBottleneck, dyn_conv_fn=dyn_conv_fn, layers=[3, 4, 6, 3],  **kwargs)
    mlog(_logger, 'model_args: {}'.format(model_args))
    model = _create_ctx_resnet('ctx_r50', pretrained, **model_args)
    if pretrained:
        url = model_urls['ctx_r50']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def ctx_r34(pretrained=False, **kwargs):
    """Constructs a Contextual ResNet-34 model.
    """
    # Configure conv functions in DynamicBottleneck
    dyn_conv_fn = CONV_FNS_MAP[kwargs['conv_fn']]    
    del kwargs['conv_fn']
        
    model_args = dict(dyn_block=DynamicBasicBlock, dyn_conv_fn=dyn_conv_fn, layers=[3, 4, 6, 3],  **kwargs)
    mlog(_logger, 'model_args: {}'.format(model_args))
    
    model = _create_ctx_resnet('ctx_r34', pretrained, **model_args)
    if pretrained:
        url = model_urls['ctx_r34']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)

    return model



@register_model
def ctx_r18(pretrained=False, **kwargs):
    """Constructs a Contextual ResNet-18 model.
    """
    # Configure conv functions in DynamicBottleneck
    dyn_conv_fn = CONV_FNS_MAP[kwargs['conv_fn']]    
    del kwargs['conv_fn']
        
    model_args = dict(dyn_block=DynamicBasicBlock, dyn_conv_fn=dyn_conv_fn, layers=[2, 2, 2, 2],  **kwargs)
    mlog(_logger, 'model_args: {}'.format(model_args))
    
    model = _create_ctx_resnet('ctx_r18', pretrained, **model_args)
    if pretrained:
        url = model_urls['ctx_r18']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
        
    return model
