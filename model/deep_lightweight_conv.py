import torch.nn as nn
import torch
import torch.nn.functional as F

class LightweightConv1d(nn.Module):
    """
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        stride=1,
        dilation=1,
        padding=0,
        num_heads=1,
        weight_softmax=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, -1)
    
        return output
    

def light_conv(in_channel, kernel_size, stride=1, dilation=1, padding=0):
    return nn.Sequential(
            LightweightConv1d(in_channel, kernel_size, stride=stride, dilation=dilation, padding=padding, num_heads=in_channel),
            nn.ReLU(inplace=True)
        )

# min_src_len=52
def dl_conv(in_channel, stride=3):
    deep_light_conv = nn.Sequential(
                              light_conv(in_channel, 3),
                              light_conv(in_channel, 7),
                              light_conv(in_channel, 15),
                              light_conv(in_channel, 30, stride=stride),
                             )
    return deep_light_conv