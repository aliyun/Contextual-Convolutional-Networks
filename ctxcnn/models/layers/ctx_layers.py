# -*- coding: utf-8 -*-
'''
Author             : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Date               : 
Last Modified By   : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Last Modified Date : 
Description        : PyTorch implementation of Contextual CNN layers (contextual convolutions, merging, visual projection)
-------- 
Copyright (c) 2022 Alibaba Inc. 
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from typing import Optional

import logging
_logger = logging.getLogger('train')
from timm.utils import mlog, log_grad, AverageMeter

from torchvision.ops import deform_conv2d as tv_deform_conv2d
# from mmcv.ops import deform_conv2d, modulated_deform_conv2d
# import ctxconv # cuda impl

from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops


soff_init_factor = 0.
woff_init_factor = 0.

"""
Merging
"""
class NonlinearMeanPool1d(nn.Module):
    # num_classes-> input classes
    def __init__(self, num_classes, embed_dim, output_dim=None, merge_norm_fn=nn.BatchNorm1d):
        super(NonlinearMeanPool1d, self).__init__()
        hidden_dim = max(embed_dim//4, 128)
        self.in_nonlinear = nn.Sequential(*[  
            nn.Linear(embed_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)])
        self.out_nonlinear = nn.Sequential(*[  
            nn.Linear(hidden_dim, output_dim), 
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        ])
        
    def forward(self, x):
        assert(len(x.size())==3), 'the shape of x should be (BatchSize, #Tokens/#Classes, #Channels)'
        x = self.in_nonlinear(x)  # (B,L,C)
        x = x.mean(dim=1) # (B,L,C) -> (B,C)
        x = self.out_nonlinear(x) # (B,C)
        return x # (B,C)


"""
Projection
"""
class VisualProjectHead(nn.Module):
    def __init__(self, spacial_dim, embed_dim, hidden_dim=512, output_dim=None, pool_sz=1, norm_fn=nn.BatchNorm1d, **kwargs):
        super(VisualProjectHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_sz)
        self.nonlinear = nn.Sequential(*[
            nn.Linear(embed_dim *pool_sz *pool_sz, hidden_dim),
            norm_fn(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
            ])

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.nonlinear(x)
        return x


# deprecated
class ClassProjectHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, output_dim: int = None):
        super(ClassProjectHead, self).__init__()
        self.nonlinear = nn.Sequential(*[  
            nn.Linear(embed_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
            ])

    def forward(self, x):
        assert(len(x.size())==3), 'the shape of x should be (BatchSize, #Tokens/#Classes, #Channels)'
        x = self.nonlinear(x) 
        return x # (B,N,C)


"""
Contextual Convolution Convolutions
"""
class StaticConv2d(nn.Module):
    """ A warpper for nn.Conv2d. """
    def __init__(
        self,
        class_emb_dim,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        norm_fn=None,
        dtype=None,
        include_context_for_offset=False,
        use_mask=True,
        writer=None
    ) -> None:
        super(StaticConv2d, self).__init__()
        self.include_context_for_offset = include_context_for_offset
        self.use_mask = use_mask
        self.norm_fn = norm_fn

        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode)
        self.depthwise = (out_channels == groups)
        mlog(_logger, 'use depthwise mode={}'.format(self.depthwise))

        if self.conv.weight.shape[1] == 1: # e.g., depthwise con
            self.weight_shape = tuple(self.conv.weight.shape)
        else:
            self.weight_shape = (1,) + tuple(self.conv.weight.shape[1:]) # (Ci,kh,kw)

        
        # Configure Dimensions
        self.total_size = reduce(lambda a,b: a*b, self.weight_shape)
        if out_channels>in_channels: # first basic-blcok for each stage
            self.context_dim = out_channels//2
        elif out_channels==in_channels: # first bottleneck/convnext-blcok for each stage
            self.context_dim = in_channels//2
        else:
            print('in_channels', in_channels, 'out_channels', out_channels)
            self.context_dim = 0
            # raise NotImplementedError
            
        if include_context_for_offset:
            self.spatial_offset_in_dim = self.weight_offset_in_dim = in_channels + self.context_dim
        else:
            self.spatial_offset_in_dim = self.weight_offset_in_dim = in_channels

        hid_factor = (kernel_size // 3) ** 2
        hid_factor = max(1, hid_factor)
        self.hidden_dim = in_channels//(4*hid_factor) # Follow the expansion of ResNet's Bottleneck, hid_factor to balance large kernels

        # handle the imbalance between in_channel and ctx
        self.spatial_offset_sz =  3*kernel_size*kernel_size if self.use_mask else 2*kernel_size*kernel_size

        self.writer = writer
        self.global_step = 0
        self.count = 0
        self.write_freq = 100000
        self.disable_weight_offset = False


    def init_weights(self):
        use_soff_flag = getattr(self, 'spatial_offset_layer', None) is not None
        use_woff_flag = getattr(self, 'weight_offset_layer', None) is not None

        if not use_soff_flag and not use_woff_flag:
            return

        if use_soff_flag:
            if self.use_mask:
                if soff_init_factor>0:
                    mlog(_logger, 'Initialization factor={} for Spatial Offsets'.format(soff_init_factor))
                    torch.nn.init.kaiming_uniform_(self.spatial_offset_layer[-1].weight, a=math.sqrt(6/soff_init_factor - 1)) # when 1, default
                else:
                    mlog(_logger, 'Zero initialization for Spatial Offsets')
                    self.spatial_offset_layer[-1].weight.data.zero_()
                    
                self.spatial_offset_layer[-1].bias.data.zero_()
            else:
                if soff_init_factor>0:
                    mlog(_logger, 'Initialization factor={} for Spatial Offsets'.format(soff_init_factor))
                    nn.init.kaiming_uniform_(self.spatial_offset_layer[-1].weight, a=math.sqrt(6/soff_init_factor - 1)) # when 1, default
                else:
                    mlog(_logger, 'Zero initialization for Spatial Offsets Without Mask')
                    self.spatial_offset_layer[-1].weight.data.zero_()
                self.spatial_offset_layer[-1].bias.data.zero_()
        
        if use_woff_flag:
            if woff_init_factor>0:
                mlog(_logger, 'Initialization factor={} for Weight Offsets'.format(woff_init_factor))
                nn.init.kaiming_uniform_(self.weight_offset_layer[-1].weight, a=math.sqrt(6/woff_init_factor - 1)) 
            else:
                mlog(_logger, 'Zero initialization for Weight Offsets')
                self.weight_offset_layer[-1].weight.data.zero_()
            self.weight_offset_layer[-1].bias.data.zero_() 

    def fast_depthwise_deformable_conv(self, x, offset, weight, mask=None, squeeze_batch=False):
        B = x.shape[0]
        # lead to much faster computation (10x), but increase memory cost
        pad_conv_weight = getattr(self, 'pad_conv_weight', None)
        if pad_conv_weight is None:
            Co,Ci,kh,kw = self.conv.weight.shape
            assert(Co==self.conv.groups and Ci==1), 'for depthwise conv only'
            mult = B if squeeze_batch else 1
            self.pad_conv_weight = torch.zeros(Co*mult,Co,kh,kw).to(x.device)
            self.scatter_idx = torch.arange(Co).to(x.device).view(-1,1,1,1).repeat(mult,1,1,1).expand(-1,-1,kh,kw)
            pad_conv_weight = self.pad_conv_weight

        pad_conv_weight.zero_()
        pad_conv_weight.scatter_(dim=1, index=self.scatter_idx, src=weight)

        if squeeze_batch:
            bias = self.conv.bias.repeat(B) if self.conv.bias is not None else self.conv.bias
            x = tv_deform_conv2d(
                x.view((1,-1)+x.shape[-2:]),
                offset.view((1,-1)+offset.shape[-2:]),
                weight=pad_conv_weight.view((-1,)+pad_conv_weight.shape[-3:]),
                bias=bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                mask=mask.view((1,-1) + mask.shape[-2:]) if mask is not None else mask
            )
            x = x.view((B,-1,) + x.shape[-2:])
            
        else:
            x = tv_deform_conv2d(
                x,
                offset,
                weight=pad_conv_weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                mask=mask
            )

        return x

    def forward(self, x, context:Optional[torch.Tensor] = None):
        return self.conv(x)


class WeightOffsetConv2d(StaticConv2d):
    """ 
        Predict Weight Offset Only. 

        The param ``dyn_group'' (G) controls the degree of flexiblity of weight offset:
            G x (in_channels x kernel_size x kernel_size)
    """
    def __init__(self, **kwargs) -> None:
        super(WeightOffsetConv2d, self).__init__(**kwargs)

        self.weight_offset_layer = nn.Sequential(*[
            nn.Linear(self.weight_offset_in_dim, self.hidden_dim), # in_channels + ctx_dim ->  hid_dim
            self.norm_fn[0](self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.total_size) # # total_size = Ci x kh x kw
        ])
        self.init_weights()

    def forward(self, x, context:Optional[torch.Tensor] = None):
        # x = x.contiguous() 
        if context is not None and self.include_context_for_offset == True:
            # assert(len(context.size())==2 )
            use_ctx_flag = True
        else:
            # used when r50 w/ weight pred
            use_ctx_flag = False

        B,_,H,W = x.size()
        shape = self.weight_shape

        if not use_ctx_flag:
            weight_x = F.adaptive_avg_pool2d(x,1).view(B,-1)
        else:
            weight_x = torch.cat([context, F.adaptive_avg_pool2d(x,1).view(B,-1)], dim=-1) # vector, non-linear fc

        # Weight Offset
        w = self.conv.weight.unsqueeze(0) #  (Co, Ci, kh, kw) -> (B, Co, Ci, kh, kw)
        if not self.disable_weight_offset:
            delta_w = self.weight_offset_layer(weight_x).view((B,)+shape) # (B, 1, Ci, kh/or/1, kw/or/1)
            w = w + delta_w #  (1, Co, Ci/G, kh, kw) +  (B, 1, Ci/G, kh, kw) -> (B, Co, Ci/G, kh, kw)
        
        # Dynamic Conv via Grouped Conv
        bias = self.conv.bias.repeat(B) if self.conv.bias is not None else self.conv.bias
        x = F.conv2d(
            x.view(1,-1,H,W), # (1, B*Ci, h, w)
            w.view((-1,)+w.shape[-3:]), # (B*Co, Ci/G, kh, kw)
            bias=bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=B*self.conv.groups
        )
        x = x.view((B,-1)+x.size()[-2:])
        
        return x
        
class SpatialOffsetConv2d(StaticConv2d):
    """
        Predict Spatial Offset and Weight Offset Simulteneously.

        use mmcv.deform_conv2d from https://github.com/open-mmlab/mmcv/blob/c60a17b6036a9ff01314a92624b7c1df4633645d/mmcv/ops/deform_conv.py#L189
        
        use mmcv.modulated_deform_conv2d from 
        https://github.com/open-mmlab/mmcv/blob/c60a17b6036a9ff01314a92624b7c1df4633645d/mmcv/ops/modulated_deform_conv.py#L153

        example:
        https://github.com/open-mmlab/mmcv/blob/c60a17b6036a9ff01314a92624b7c1df4633645d/mmcv/ops/saconv.py

    """
    def __init__(self, **kwargs) -> None:
        super(SpatialOffsetConv2d, self).__init__(**kwargs)
        
        self.spatial_offset_layer = nn.Sequential(*[
            nn.Conv2d(self.spatial_offset_in_dim, self.hidden_dim, 1, stride=1, padding=0, dilation=1, bias=True), # 1x1 bottleneck conv: in_channels + ctx_dim ->  hid_dim
            self.norm_fn[1](self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.spatial_offset_sz, self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, bias=True), # 3x3 conv: hid_dim -> 3*kh*kw
        ])
        
        self.init_weights()
    
    def forward(self, x, context:Optional[torch.Tensor] = None):
        """
            x: previous-layer features
            context: merged class embeddings
        """
        if context is not None and self.include_context_for_offset == True:
            assert(len(context.size())==2 )
            use_ctx_flag = True
        else:
            # used when r50 w/ weight pred
            assert(len(x.size())==4)
            use_ctx_flag = False

        B,_,H,W = x.size()

        if not use_ctx_flag:
            # For Baseline
            spatial_x = x
        else:
            # with ctx as extra input
            spatial_x = torch.cat([context.view(B,-1,1,1).expand((-1,-1)+x.size()[-2:]),x], dim=1) # feature map, non-linear conv
        
        
        # Spatial Offsets and Mask （dcnv1,） (dcnv2)
        out = self.spatial_offset_layer(spatial_x) # (B,3*kh*kw,h,w)
        if self.use_mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1) # x,y offsets, (B,2*kh*kw,h,w)
            mask = torch.sigmoid(mask) # mask, (B,1*kh*kw,h,w)
        else:
            offset = out
            mask = None
        
        x = tv_deform_conv2d(
            x,
            offset,
            weight=self.conv.weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            mask=mask if mask is not None else None
        )
            
        return x


class ContextualConv2d(StaticConv2d):
    """
        Predict Spatial Offset and Weight Offset Simulteneously.

        use mmcv.deform_conv2d from https://github.com/open-mmlab/mmcv/blob/c60a17b6036a9ff01314a92624b7c1df4633645d/mmcv/ops/deform_conv.py#L189

        use mmcv.modulated_deform_conv2d from 
        https://github.com/open-mmlab/mmcv/blob/c60a17b6036a9ff01314a92624b7c1df4633645d/mmcv/ops/modulated_deform_conv.py#L153

        example:
        https://github.com/open-mmlab/mmcv/blob/c60a17b6036a9ff01314a92624b7c1df4633645d/mmcv/ops/saconv.py

    """
    def __init__(self, **kwargs) -> None:
        super(ContextualConv2d, self).__init__(**kwargs)
        if not self.use_mask:
            assert(bias is False), 'fixme: `deform_conv2d` op is not available when bias is True'

        self.spatial_offset_layer = nn.Sequential(*[
            nn.Conv2d(self.spatial_offset_in_dim, self.hidden_dim, 1, stride=1, padding=0, dilation=1, bias=True), # 1x1 bottleneck conv: in_channels + ctx_dim ->  hid_dim
            self.norm_fn[1](self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.spatial_offset_sz, self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, bias=True), # 3x3 conv: hid_dim -> 3*kh*kw
        ])
        
        self.weight_offset_layer = nn.Sequential(*[
            nn.Linear(self.weight_offset_in_dim, self.hidden_dim), # in_channels + ctx_dim ->  hid_dim
            self.norm_fn[0](self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.total_size) # # total_size = Ci x kh x kw
        ])

        self.init_weights()  

    def forward(self, x, context:Optional[torch.Tensor] = None):
        """
            x: previous-layer features
            context: merged class embeddings
        """
        # x = x.contiguous() 
        if context is not None and self.include_context_for_offset == True:
            assert(len(context.size())==2 )
            use_ctx_flag = True
        else:
            # used when r50 w/ weight pred
            assert(len(x.size())==4)
            use_ctx_flag = False

        B,_,H,W = x.size()
        shape = self.weight_shape

        if not use_ctx_flag:
            # For Baseline
            spatial_x = x
            weight_x = F.adaptive_avg_pool2d(x,1).view(B,-1)
        else:
            # With context as extra input
            spatial_x = torch.cat([context.view(B,-1,1,1).expand((-1,-1)+x.size()[-2:]),x], dim=1) # feature map, non-linear conv
            weight_x = torch.cat([context, F.adaptive_avg_pool2d(x,1).view(B,-1)], dim=-1) # vector, non-linear fc

        # Spatial Offsets and Mask （dcnv1,） (dcnv2)
        out = self.spatial_offset_layer(spatial_x) # (B,3*kh*kw,h,w)
        if self.use_mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1) # x,y offsets, (B,2*kh*kw,h,w)
            mask = torch.sigmoid(mask) # mask, (B,1*kh*kw,h,w)
        else:
            offset = out
            mask = None

        # Weight Offset
        w = self.conv.weight.unsqueeze(0).expand(B,-1,-1,-1,-1) #  (Co, Ci, kh, kw) -> (B, Co, Ci, kh, kw)
        if not self.disable_weight_offset:
            delta_w = self.weight_offset_layer(weight_x).view((B,)+shape) # (B, 1, Ci, kh, kw)
            w = w + delta_w #  (B, Co, Ci, kh, kw) +  (B, 1, Ci, kh, kw) -> (B, Co, Ci, kh, kw)

        conv = tv_deform_conv2d
        oh, ow = offset.size()[-2:] # could be different from (H,W) when stride>1
        x = conv(
            x.view(1,-1,H,W),
            offset.view(1,-1,oh,ow), 
            weight=w.view((-1,)+w.shape[-3:]),
            bias=self.conv.bias.repeat(B) if self.conv.bias is not None else self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            mask=mask.view(1,-1,oh,ow) if mask is not None else mask, # (B,1*kh*kw,h,w) -> (1,B*1*kh*kw,h,w)
        )
        x = x.view((B,-1) + x.shape[-2:])
        
        return x

    

# CUDA implementation of ctx conv2d
# def ctx_conv2d(input, offset, weight, bias, stride, padding, dilation, mask):
#     _assert_has_ops()
#     out_channels = weight.shape[0]
#     use_mask = mask is not None

#     if mask is None:
#         mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

#     if bias is None:
#         bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

#     stride_h, stride_w = _pair(stride)
#     pad_h, pad_w = _pair(padding)
#     dil_h, dil_w = _pair(dilation)
#     weights_h, weights_w = weight.shape[-2:]
#     _, n_in_channels, _, _ = input.shape

#     n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
#     n_weight_grps = n_in_channels // weight.shape[1]

#     # print('n_weight_grps', n_weight_grps,'n_offset_grps', n_offset_grps)

#     if n_offset_grps == 0:
#         raise RuntimeError(
#             "the shape of the offset tensor at dimension 1 is not valid. It should "
#             "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
#             f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
#         )
#     return ctxconv.autocast(
#         input,
#         weight,
#         offset,
#         mask,
#         bias,
#         stride_h,
#         stride_w,
#         pad_h,
#         pad_w,
#         dil_h,
#         dil_w,
#         n_weight_grps,
#         n_offset_grps,
#         use_mask
#     )