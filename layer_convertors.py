import timm
import types
import torch
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as sm

from collections import OrderedDict

def __classinit(cls):
    return cls._init__class()

def __is_generator_empty(generator):
    try:
        next(generator)
        return False
    except StopIteration:
        return True

def convert_inplace(net, convertor):
    stack = [net]

    while stack:
        node = stack[-1]

        stack.pop()

        for name, child in node.named_children():
            if not __is_generator_empty(child.children()):
                stack.append(child)

            setattr(node, name, convertor(child))

@__classinit
class LayerConvertor(object):
    @classmethod
    def _init__class(cls):
        cls._registry = { }

        return cls()

    def __call__(self, layer):
        if type(layer) in self._registry:
            return self._registry[type(layer)](layer)
        else:
            return self._func_None(layer)

    @classmethod
    def _func_None(cls, layer):
        return layer


@__classinit
class LayerConvertorNNSPT(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            nn.Conv1d: getattr(cls, '_func_Conv1d'),
            nn.MaxPool1d: getattr(cls, '_func_MaxPool1d'),
            nn.AvgPool1d: getattr(cls, '_func_AvgPool1d'),
            nn.BatchNorm1d: getattr(cls, '_func_BatchNorm1d'),
            nn.AdaptiveAvgPool1d: getattr(cls, '_func_AdaptiveAvgPool1d')
        }

        return cls()

    @staticmethod
    def __expand_tuple(param):
        return (param[0], param[0], param[0])

    @classmethod
    def _func_None(cls, layer):
        return layer

    @classmethod
    def _func_AdaptiveAvgPool1d(cls, layer1d):
        kwargs = {
            'output_size': layer1d.output_size
        }

        layer3d = nn.AdaptiveAvgPool3d(**kwargs)
        return layer3d

    @classmethod
    def _func_Conv1d(cls, layer1d):
        kwargs = {
            'in_channels': layer1d.in_channels,
            'out_channels': layer1d.out_channels,
            'kernel_size': cls.__expand_tuple(layer1d.kernel_size),
            'stride': cls.__expand_tuple(layer1d.stride),
            'padding': cls.__expand_tuple(layer1d.padding),
            'dilation': cls.__expand_tuple(layer1d.dilation),
            'groups': layer1d.groups,
            'bias': 'bias' in layer1d.state_dict(),
            'padding_mode': layer1d.padding_mode
        }

        layer3d = nn.Conv3d(**kwargs)

        return layer3d

    @classmethod
    def _func_BatchNorm1d(cls, layer1d):
        kwargs = {
            'num_features': layer1d.num_features,
            'eps': layer1d.eps,
            'momentum': layer1d.momentum,
            'affine': layer1d.affine,
            'track_running_stats': layer1d.track_running_stats
        }

        layer3d = nn.BatchNorm3d(**kwargs)

        return layer3d

    @classmethod
    def _func_MaxPool1d(cls, layer1d):
        kwargs = {
            'kernel_size': layer1d.kernel_size,
            'stride': layer1d.stride,
            'padding': layer1d.padding,
            'dilation': layer1d.dilation,
            'return_indices': layer1d.return_indices,
            'ceil_mode': layer1d.ceil_mode
        }

        layer3d = nn.MaxPool3d(**kwargs)

        return layer3d

    @classmethod
    def _func_AvgPool1d(cls, layer1d):
        kwargs = {
            'kernel_size': layer1d.kernel_size,
            'stride': layer1d.stride,
            'padding': layer1d.padding,
            'ceil_mode': layer1d.ceil_mode,
            'count_include_pad': layer1d.count_include_pad,
            'divisor_override': layer1d.divisor_override
        }

        layer3d = nn.AvgPool3d(**kwargs)

        return layer3d

@__classinit
class LayerConvertorSm(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            sm.decoders.unet.decoder.DecoderBlock: getattr(cls, '_func_sm_unet_decoder_DecoderBlock'),
            sm.decoders.unet.decoder.UnetDecoder: getattr(cls, '_func_sm_unet_decoder_UnetDecoder'),
            sm.decoders.unetplusplus.decoder.DecoderBlock: getattr(cls, '_func_sm_unetplusplus_decoder_DecoderBlock'),
            timm.layers.grn.GlobalResponseNorm: getattr(cls, '_func_timm_GlobalResponseNorm'),
            timm.models._efficientnet_blocks.SqueezeExcite: getattr(cls, '_func_timm_SqueezeExcite'),
            timm.layers.norm_act.BatchNormAct2d: getattr(cls, '_func_timm_layers_norm_act_BatchNormAct2d'),
            timm.layers.norm.LayerNorm2d: getattr(cls, '_func_timm_layers_norm_LayerNorm2d'),
            timm.models.convnext.ConvNeXtBlock: getattr(cls, '_func_timm_models_convnext_ConvNeXtBlock'),
        }

        return cls()

    @classmethod
    def _func_timm_GlobalResponseNorm(cls, layer):
        if layer.channel_dim == -1:
            layer.spatial_dim = (1, 2, 3)
            layer.wb_shape = (1, 1, 1, 1, -1)
        else:
            layer.spatial_dim = (2, 3, 4)
            layer.wb_shape = (1, -1, 1, 1, 1)

        return layer

    @staticmethod
    def _sm_unet_decoder_decoderblock_forward(self, x, skip=None, shape=None):
        if shape is not None:
            scale_factor = list()

            getdim = lambda vector, axis : vector.shape[axis]

            naxis = len(x.shape)
            for axis in np.arange(2, naxis):
                scale_factor.append(shape[axis]/getdim(x, axis))

            scale_factor = tuple(scale_factor)
        else:
            scale_factor = 2

        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        return x

    @staticmethod
    def _sm_unet_decoder_unetdecoder_forward(self, *features):
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            if i < len(skips) - 1:
                skip = skips[i]
                shape = skips[i].shape
            else:
                skip = None
                shape = skips[i].shape

            x = decoder_block(x, skip, shape)

        return x

    @classmethod
    def _func_sm_unet_decoder_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_decoderblock_forward, layer)

        return layer

    @classmethod
    def _func_sm_unet_decoder_UnetDecoder(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_unetdecoder_forward, layer)

        return layer

    @classmethod
    def _func_sm_unetplusplus_decoder_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_decoderblock_forward, layer)

        return layer

    @staticmethod
    def _timm_squeezeexcite_forward(self, x):
        """
            :NOTE:
                it is a copy of timm.layers.squeeze_excite.SEModule function with correct operations under dims
        """
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

    @classmethod
    def _func_timm_SqueezeExcite(cls, layer):
        layer.forward = types.MethodType(cls._timm_squeezeexcite_forward, layer)

        return layer

    @staticmethod
    def _timm_layers_norm_act_batchnormact2d_forward(self, x):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = nn.functional.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        x = self.drop(x)
        x = self.act(x)

        return x

    @classmethod
    def _func_timm_layers_norm_act_BatchNormAct2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_layers_norm_act_batchnormact2d_forward, layer)

        return layer

    @staticmethod
    def _timm_layers_norm_layernorm2d_forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)

        return x

    @classmethod
    def _func_timm_layers_norm_LayerNorm2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_layers_norm_layernorm2d_forward, layer)

        return layer

    @staticmethod
    def _timm_models_convnext_convnextblock_forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 4, 1, 2, 3)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)

        return x

    @classmethod
    def _func_timm_models_convnext_ConvNeXtBlock(cls, layer):
        layer.forward = types.MethodType(cls._timm_models_convnext_convnextblock_forward, layer)

        return layer

