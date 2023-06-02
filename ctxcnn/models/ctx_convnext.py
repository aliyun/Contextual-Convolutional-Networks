# -*- coding: utf-8 -*-
'''
Author             : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Date               : 
Last Modified By   : ruihe.lsx (ruihe.lsx@alibaba-inc.com)
Last Modified Date : 2023-05-08 15:25
Description        : PyTorch implementation of Contextual ConvNeXt
                     (modified from https://github.com/huggingface/pytorch-image-models)

-------- 
Copyright (c) 2022 Alibaba Inc. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, StaticConv2d
from timm.models.registry import register_model
from .ctx_resnet import CONV_FNS_MAP, Contextualizing

import logging

_logger = logging.getLogger('train')
from timm.utils import mlog, log_grad, get_topk_predictions

__all__ = ['CtxConvNeXt']  # model_registry will add each entrypoint fn to this


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self,
                 conv_fn,
                 class_emb_dim,
                 dim,
                 width_multiplier=4,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 include_context_for_offset=True,
                 use_mask=True,
                 drop_without_ctx=True,
                 writer=None):
        super().__init__()

        if drop_without_ctx == True and conv_fn != StaticConv2d:
            drop_path = 0.
            mlog(_logger, 'Setting Contextual Block drop_path_rate to 0.')

        self.dwconv = conv_fn(
            class_emb_dim=class_emb_dim,
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            norm_fn=(nn.LayerNorm,
                     partial(LayerNorm, eps=1e-6,
                             data_format='channels_first')),
            include_context_for_offset=include_context_for_offset,
            use_mask=use_mask,
            writer=writer)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, width_multiplier *
            dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(width_multiplier * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):
        x, context = inputs
        input = x
        x = self.dwconv(x, context)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return (x, context)


class CtxConvNeXt(nn.Module):
    r""" CtxConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            dyn_conv_fn,
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            width_multipliers=[4, 4, 4, 4],
            drop_path_rate=0.,
            layer_scale_init_value=1e-6,
            head_init_scale=1.,
            ctx_layers=(1, 1, 1, 1),
            ctx_ks=(500, 200, 50, 1),
            dyn_block_freq=1,
            class_emb_dim=256,
            class_merge_fn='nonlinear_mean',
            include_context_for_offset=True,
            use_mask=True,
            use_class_proj=False,
            lv_cls_emb=False,
            logit_scale_method='fixed',
            emb_init_method='clip',
            drop_without_ctx=False,
            ctx_rank_loss=False,
            block_args=None,
            cls_args=None,
            strict_load=False,
            writer=None,
            downstream_mode=False,  # FIXME
            **kwargs  # unused parameters from timm, e.g., drop_rate
    ):
        super().__init__()
        self.num_classes = num_classes
        block_args = block_args or dict()
        self.cls_args = cls_args or dict()
        self.ctx_layers = ctx_layers
        self.ctx_ks = tuple(ctx_ks)
        self.dyn_conv_fn = dyn_conv_fn
        self.class_merge_fn = class_merge_fn
        self.strict_load = strict_load
        self.writer = writer
        mlog(_logger, 'self.ctx_layers: {}'.format(self.ctx_layers))

        self.downsample_layers = nn.ModuleList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        static_conv_fn = StaticConv2d
        first_ctx_layer = True
        self.stages = nn.ModuleList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks

        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        drop_without_ctx = drop_without_ctx if dyn_block_freq > 1 else False

        cur = 0
        for i in range(4):
            if ctx_layers[i]:
                if first_ctx_layer:
                    conv_fn = static_conv_fn
                    first_ctx_layer = False
                else:
                    conv_fn = dyn_conv_fn
            else:
                conv_fn = static_conv_fn
            stage = nn.Sequential(*[
                Block(conv_fn if j % dyn_block_freq == 0 else static_conv_fn,
                      class_emb_dim,
                      dim=dims[i],
                      width_multiplier=width_multipliers[i],
                      drop_path=dp_rates[cur + j],
                      layer_scale_init_value=layer_scale_init_value,
                      include_context_for_offset=include_context_for_offset,
                      use_mask=use_mask,
                      drop_without_ctx=drop_without_ctx,
                      writer=writer) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        mlog(_logger, self.downsample_layers)
        mlog(_logger, self.stages)

        # Head (Pooling and Classifier)
        self.num_features = num_features = dims[-1]
        self.cls_args['class_emb_dim'] = class_emb_dim
        self.cls_args['use_class_proj'] = use_class_proj
        self.cls_args['lv_cls_emb'] = lv_cls_emb
        self.cls_args[
            'include_context_for_offset'] = include_context_for_offset
        self.cls_args['logit_scale_method'] = logit_scale_method
        self.cls_args['emb_init_method'] = emb_init_method
        self.cls_args['block_expansion'] = 1
        self.cls_args['cls_norm_fn'] = nn.LayerNorm
        self.cls = Contextualizing(dims, num_classes, ctx_layers, ctx_ks,
                                   ctx_rank_loss, dyn_conv_fn, class_merge_fn,
                                   **self.cls_args)
        mlog(_logger, 'Contextualizing:', self.cls)

        if downstream_mode:
            self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)
            mlog(_logger, 'Downstream Mode=True! Enable self.fc:', self.fc)

        self.init_weights()
        self.visualization = False
        self.single_output = False
        self.intermediate_featrues = []

    def no_weight_decay(self):
        return (
            'cls.logit_scale',
            'cls.logit_scale.0',
            'cls.logit_scale.1',
            'cls.logit_scale.2',
            'cls.logit_scale.3',
            'cls.class_token_embedding.weight',
        )

    def emb_filter(self):
        return ('cls.class_token_embedding.weight', )

    def load_state_dict(self,
                        state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        init_state_dict = self.state_dict()
        replacements = []
        adds = []
        mismatches = []
        strict = strict and self.strict_load
        # Fixme
        static_to_dynamic_rules = {
            'dwconv.weight': 'dwconv.conv.weight',
            'dwconv.bias': 'dwconv.conv.bias',
        }
        for k, v in init_state_dict.items():
            if k not in state_dict.keys():
                # check whether the corresponding static weights exists
                not_find_replacement_flag = True
                for old, new in static_to_dynamic_rules.items():
                    static_k = k.replace(new, old)
                    if static_k in state_dict.keys():
                        if state_dict[static_k].shape == init_state_dict[
                                k].shape:
                            state_dict[k] = state_dict[
                                static_k]  # rename static weight with new name
                            replacements.append(k)
                            del state_dict[static_k]
                        else:
                            if not strict:  # which won't throw exception
                                mismatches.append(k)
                                state_dict[k] = init_state_dict[
                                    k]  # use random init weights
                        not_find_replacement_flag = False
                        break

                # not find any replacement, then denoted by new weights
                if not_find_replacement_flag:
                    adds.append(k)
                    state_dict[k] = init_state_dict[
                        k]  # use random init weights

            else:
                if not strict:  # which won't throw exception
                    if state_dict[k].shape != init_state_dict[k].shape:
                        mismatches.append(k)
                        state_dict[k] = init_state_dict[
                            k]  # use random init weights

        mlog(_logger, 'Find Replacements Weights: {}'.format(replacements))
        mlog(_logger, 'Newly Added Weights: {}'.format(adds))
        mlog(
            _logger,
            'Mismatched Weights (Used only when not strict): {}'.format(
                mismatches))

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
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
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
                    0, 'Missing key(s) in state_dict: {}. '.format(', '.join(
                        '"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__, "\n\t".join(error_msgs)))
        return torch.nn.modules.module._IncompatibleKeys(
            missing_keys, unexpected_keys)

    def init_weights(self, head_init_scale=1.0):
        for n, m in self.named_modules():
            if 'offset' in n:
                continue
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):  # fix 0514
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        flag1, flag2, flag3, flag4 = self.ctx_layers
        n2, n3, n4, _ = self.ctx_ks
        ds1, ds2, ds3, ds4 = self.downsample_layers
        stage1, stage2, stage3, stage4 = self.stages
        y1, y2, y3, y4 = [
            None,
        ] * 4
        B = x.size(0)

        c = self.cls.generate(device=x.device).unsqueeze(0).expand(B, -1, -1)
        ctx = None
        x = ds1(x)
        x, _ = stage1((x, ctx))
        if flag1:
            y1, c = self.cls(x, c, stage_idx=0)
            ctx = self.cls.merge(c, 1, n2)
        x = ds2(x)
        x, _ = stage2((x, ctx))

        if flag2:
            y2, c = self.cls(x, c, stage_idx=1)
            ctx = self.cls.merge(c, 2, n3)
        x = ds3(x)
        x, _ = stage3((x, ctx))

        if flag3:
            y3, c = self.cls(x, c, stage_idx=2)
            ctx = self.cls.merge(c, 3, n4)
        x = ds4(x)
        x, _ = stage4((x, ctx))

        return x.mean([-2, -1])

    def forward(self, x):
        flag1, flag2, flag3, flag4 = self.ctx_layers
        n2, n3, n4, _ = self.ctx_ks
        ds1, ds2, ds3, ds4 = self.downsample_layers
        stage1, stage2, stage3, stage4 = self.stages
        y1, y2, y3, y4 = [
            None,
        ] * 4
        B = x.size(0)
        c = self.cls.generate(device=x.device).unsqueeze(0).expand(B, -1, -1)

        ctx = None
        x = ds1(x)
        x, _ = stage1((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        if flag1:
            y1, c = self.cls(x, c, stage_idx=0)
            ctx = self.cls.merge(c, 1, n2)
        x = ds2(x)
        x, _ = stage2((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        if flag2:
            y2, c = self.cls(x, c, stage_idx=1)
            ctx = self.cls.merge(c, 2, n3)
        x = ds3(x)
        x, _ = stage3((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        if flag3:
            y3, c = self.cls(x, c, stage_idx=2)
            ctx = self.cls.merge(c, 3, n4)
        x = ds4(x)
        x, _ = stage4((x, ctx))
        if self.visualization and not self.single_output:
            x.register_hook(lambda g: self.intermediate_featrues.append(g))

        y4, c = self.cls(x, c, stage_idx=3)

        if self.visualization and self.single_output:
            vis_y, _ = get_topk_predictions([y1, y2, y3, y4], self.ctx_ks)
            return vis_y
        return [y1, y2, y3, y4]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k":
    "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def ctx_convnext_tiny(pretrained=False, **kwargs):
    dyn_conv_fn = CONV_FNS_MAP[kwargs['conv_fn']]
    del kwargs['conv_fn']
    model_args = dict(dyn_conv_fn=dyn_conv_fn, **kwargs)
    model = CtxConvNeXt(depths=[3, 3, 9, 3],
                        dims=[96, 192, 384, 768],
                        **model_args)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu",
                                                        check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def ctx_convnext_small(pretrained=False, **kwargs):
    dyn_conv_fn = CONV_FNS_MAP[kwargs['conv_fn']]
    del kwargs['conv_fn']
    model_args = dict(dyn_conv_fn=dyn_conv_fn, **kwargs)
    model = CtxConvNeXt(depths=[3, 3, 27, 3],
                        dims=[96, 192, 384, 768],
                        **model_args)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def ctx_convnext_base(pretrained=False, **kwargs):
    dyn_conv_fn = CONV_FNS_MAP[kwargs['conv_fn']]
    del kwargs['conv_fn']
    model_args = dict(dyn_conv_fn=dyn_conv_fn, **kwargs)
    model = CtxConvNeXt(depths=[3, 3, 27, 3],
                        dims=[128, 256, 512, 1024],
                        **model_args)
    if pretrained:
        url = model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
