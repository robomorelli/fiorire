from typing import Union, Tuple

from torch import nn
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict

clip_x_to0 = 1e-4

def SmashTo0(x):
    return 0*x

class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value=5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return 1. + self.min_value \
               + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1 + (x * x))))

class ClippedTanh(nn.Module):

    def __init__(self, min_value=5e-3):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.999 * torch.tanh(x))

class SmashTo0(nn.Module):

    def __init__(self):
        super(SmashTo0, self).__init__()

    def forward(self, x):
        return 0*x

class h1_prior(nn.Module):
    def __init__(self, in_features, out_features):
        super(h1_prior, self).__init__()
        self.h1_prior_w = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.h1_prior_b = nn.Parameter(torch.ones(out_features), requires_grad=False)

    def forward(self, x):
        x = F.linear(x, self.h1_prior_w, self.h1_prior_b)
        return x

#The usual way to do this is to use the functional interface to redefine the forward

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ConstrainedDec(nn.Linear):
    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class Dec1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dec1, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x



def conv_block1D(
        in_f,
        out_f,
        kernel_size=3,
        padding=1,
        dilation=1,
        activation=nn.ReLU(),
        batch_norm=True,
        pool=True,
        pool_ks=2,
        pool_stride=2,
        pool_pad=0,
        *args, **kwargs
    ):
    """
    Conv1d block mirroring conv_block (2D version) for consistency.
    """
    # Ensure parameters are ints
    if isinstance(pool_ks, (tuple, list)):
        pool_ks = pool_ks[0]
    if isinstance(pool_stride, (tuple, list)):
        pool_stride = pool_stride[0]
    if isinstance(pool_pad, (tuple, list)):
        pool_pad = pool_pad[0]
    if isinstance(padding, (tuple, list)):
        padding = padding[0]

    layers = [nn.Conv1d(
        in_f, out_f,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        *args, **kwargs
    )]

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_f))

    if activation:
        layers.append(activation)

    if pool:
        layers.append(nn.MaxPool1d(kernel_size=pool_ks, stride=pool_stride, padding=pool_pad))



    return nn.Sequential(*layers)


def deconv_block1D(
        in_f,
        out_f,
        kernel_size=2,
        stride=2,
        dilation=1,
        output_padding=None,
        activation=nn.ReLU(),
        batch_norm=True,
        double_deconv=False,
        conv_kernel_size=3,
        conv_padding=0,
        conv_stride=1,
        conv_dilation=1,
        *args, **kwargs
    ):
    """
    ConvTranspose1d block mirroring deconv_block (2D version).
    """
    if isinstance(kernel_size, (tuple, list)):
        kernel_size = kernel_size[0]
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(output_padding, (tuple, list)):
        output_padding = output_padding[0]
    if output_padding is None:
        output_padding = 0

    layers = [nn.ConvTranspose1d(
        in_f, out_f,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        dilation=dilation,
        *args, **kwargs
    )]

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_f))
    if activation:
        layers.append(activation)

    if double_deconv:
        layers.append(nn.Conv1d(out_f, out_f,
                                kernel_size=conv_kernel_size,
                                stride=conv_stride,
                                padding=conv_padding,
                                dilation=conv_dilation,
                                *args, **kwargs))

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_f))
    if activation:
        layers.append(activation)

    return nn.Sequential(*layers)




def conv_block(
        in_f,
        out_f,
        kernel_size=3,
        padding=1,
        dilation=1,
        activation=nn.ReLU(),
        batch_norm=True,
        pool=True,
        pool_ks=2,
        pool_stride=2,
        pool_pad=0,
        *args, **kwargs
    ):
    """
    Conv2d block with optional BatchNorm, activation, and pooling.
    Supports pool_ks, pool_stride, pool_pad as int or tuple (H, W).
    """
    # Ensure pool parameters are tuples
    if isinstance(pool_ks, int):
        pool_ks = (pool_ks, pool_ks)
    if isinstance(pool_stride, int):
        pool_stride = (pool_stride, pool_stride)
    if isinstance(pool_pad, int):
        pool_pad = (pool_pad, pool_pad)
    if isinstance(padding, int):
        padding = (padding, padding)

    layers = [nn.Conv2d(
        in_f, out_f,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        *args, **kwargs
    )]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_f))

    if activation:
        layers.append(activation)

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=pool_ks,
            stride=pool_stride,padding=pool_pad))


    return nn.Sequential(*layers)

''' 
def bottleneck1D(in_channels, out_channels, activation=nn.ReLU(), bottleneck_activation=nn.ReLU(), batch_norm=True):
    """
    Crea un bottleneck 1D con conv 1x1, optional BatchNorm e attivazione.
    """
    layers = OrderedDict()
    layers["bottleneck_conv"] = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    if batch_norm:
        layers["bottleneck_bn"] = nn.BatchNorm1d(out_channels)
    if bottleneck_activation is not None:
        layers["act1"] = bottleneck_activation
    else:
        layers["act1"] = activation
    return nn.Sequential(layers)
'''

def bottleneck1D(bottleneck_conv, activation=nn.ReLU(),
                 flattened=True, flattened_size=None, latent_dim=None,
                 bottleneck_activation=nn.ReLU(), batch_norm=True):
    """
    Crea un bottleneck 1D con conv 1x1, optional BatchNorm e attivazione.
    """
    layers = OrderedDict()
    layers["bottleneck_conv"] = bottleneck_conv
    if batch_norm:
        layers["bottleneck_bn"] = nn.BatchNorm1d(bottleneck_conv.out_channels)
    if activation is not None:
        layers["act1"] = activation

    if flattened:
        layers["flatten"] = nn.Flatten()
        layers["to_latent"] = nn.Linear(flattened_size, latent_dim)
        if batch_norm is not None:
            layers["batch_norm_latent"] = nn.BatchNorm1d(latent_dim)
        if bottleneck_activation is not None:
            layers["act"] = bottleneck_activation

    else:
        raise NotImplementedError("Non-flattened bottleneck not implemented yet.")



    return nn.Sequential(layers)



def bottleneck2D(bottleneck_conv,
                 flattened=True, flattened_size=None, latent_dim=None,
                 activation=nn.ReLU(), bottleneck_activation=nn.ReLU(), batch_norm=True):
    """
    Crea un bottleneck 2D: Flatten → Dense → Dense → Unflatten.
    """

    layers = OrderedDict()

    layers['bottleneck_conv'] = bottleneck_conv

    if batch_norm:
        layers['batch_norm_conv'] = nn.BatchNorm2d(bottleneck_conv.out_channels)
    if activation is not None:
        layers['activation'] = activation

    if flattened:
        layers["flatten"] = nn.Flatten()
        layers["to_latent"] = nn.Linear(flattened_size, latent_dim)
        if batch_norm is not None:
            layers["batch_norm_latent"] = nn.BatchNorm1d(latent_dim)
        if bottleneck_activation is not None:
            layers["act1"] = bottleneck_activation

    else:
        raise NotImplementedError("Non-flattened bottleneck not implemented yet.")

    return nn.Sequential(layers)


def deconv_block(in_f, out_f,
                 kernel_size=2, stride=2, dilation=1, output_padding=None,
                 activation=nn.ReLU(), batch_norm=True, double_deconv=False,
                 conv_kernel_size=3, conv_padding=0, conv_stride=1, conv_dilation=1,
                 reshape=False, flattened=True, latent_dim=None, flattened_size=None,
                 first_deconv_channels=None, h_enc=None, w_enc=None,
                 *args, **kwargs):
    """
    ConvTranspose2d block with optional reshape prepending, BatchNorm and activation.

    Args:
        in_f: Input channels for ConvTranspose2d
        out_f: Output channels for ConvTranspose2d
        kernel_size: Kernel size for ConvTranspose2d
        stride: Stride for ConvTranspose2d
        dilation: Dilation for ConvTranspose2d
        output_padding: Output padding for ConvTranspose2d
        activation: Activation function (default: ReLU)
        batch_norm: Whether to use BatchNorm
        double_deconv: Whether to add refinement Conv2d after ConvTranspose2d
        conv_kernel_size: Kernel size for refinement conv
        conv_padding: Padding for refinement conv
        conv_stride: Stride for refinement conv
        conv_dilation: Dilation for refinement conv
        reshape: If True, prepend reshape operations (Linear → Unflatten → Conv)
        flattened: Whether input is flattened (used when reshape=True)
        latent_dim: Latent dimension size (used when reshape=True)
        flattened_size: Size after Linear layer (used when reshape=True)
        first_deconv_channels: Channels after unflattening (used when reshape=True)
        h_enc: Height after unflattening (used when reshape=True)
        w_enc: Width after unflattening (used when reshape=True)
    """
    # Ensure kernel_size, stride, output_padding are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if output_padding is None:
        output_padding = (0, 0)
    elif isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    layers = OrderedDict()

    # Prepend reshape operations if requested
    if reshape:
        if flattened:
            layers['latent_to_flatten'] = nn.Linear(latent_dim, flattened_size)
            if batch_norm:
                layers['batch_norm_1d'] = nn.BatchNorm1d(flattened_size)
            if activation is not None:
                layers['act_reshape'] = activation
            layers['unflatten'] = nn.Unflatten(1, (first_deconv_channels, h_enc, w_enc))


    # Main ConvTranspose2d layer
    layers['deconv'] = nn.ConvTranspose2d(
        in_f, out_f,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        dilation=dilation,
        *args, **kwargs
    )

    # BatchNorm and activation after deconv
    if batch_norm:
        layers['batch_norm'] = nn.BatchNorm2d(out_f)
    if activation is not None:
        layers['activation'] = activation

    # Optional refinement conv
    if double_deconv:
        layers['refine_conv'] = nn.Conv2d(
            out_f, out_f,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
            stride=conv_stride,
            dilation=conv_dilation
        )

    # BatchNorm and activation after deconv
    if batch_norm:
        layers['batch_norm'] = nn.BatchNorm2d(out_f)
    if activation is not None:
        layers['activation'] = activation

    return nn.Sequential(layers)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.00005):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class LinConstr(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinConstr, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0*10**6, max=1.0*10**6), self.bias)
        return x
'''
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
'''
