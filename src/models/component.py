import torch
from torch import nn
from math import sqrt
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class AxialAttention(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1):
        super().__init__()

        # Shared projections
        self.proj1 = nn.Linear(in_dim, out_dim, bias=False)
        self.proj2 = nn.Linear(in_dim, out_dim, bias=False)
        self.proj3 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x_h = rearrange(x, "b c h w -> (b w) h c")
        x1_h = self.proj1(x_h)
        x2_h = self.proj2(x_h)
        x3_h = self.proj3(x_h)
        z1 = F.scaled_dot_product_attention(x1_h, x2_h, x3_h)  # Output: (B * W, H, C)
        z1 = rearrange(z1, "(b w) h c -> b c h w", b=b, w=w)

        x_w = rearrange(x, "b c h w -> (b h) w c")
        x1_w = self.proj1(x_w)
        x2_w = self.proj2(x_w)
        x3_w = self.proj3(x_w)
        z2 = F.scaled_dot_product_attention(x1_w, x2_w, x3_w)  # Output: (B * H, W, C)
        z2 = rearrange(z2, "(b h) w c -> b c h w", b=b, h=h)

        return z1 + z2


class ImagePCA(nn.Module):
    def __init__(self, top_k: int = 4, mu: float = 1e-4):
        super().__init__()
        self.k = top_k
        self.mu = mu

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.double()
            X = rearrange(x, "b 1 h w -> b h w")
            gram = torch.bmm(X.permute(0, 2, 1), X)

            muI = (
                self.mu
                * torch.diag(X.mean(0)).mean()
                * torch.eye(n=X.shape[-1], device=X.device)
            )
            _, vecs = torch.linalg.eigh(
                gram + muI
            )  # Tikhonov regularization to prevent ill-condition

            # vecs = vecs[..., : self.k]  # bottom-k
            vecs = torch.flip(vecs, dims=(-1,))[..., : self.k]  # top-k

            z = rearrange(torch.bmm(X, vecs), "b h k -> b (h k)").float()

        return z.to(x.device)


class ConvSum2d(nn.Module):
    """A custom PyTorch 2D convolution module designed to perform operations

    across summed input channels and weights based on the specified mode.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of
          the input. Default: 0.
        bias (bool, optional): If True, adds a learnable bias to the output.
          Default: True.
        mode (str, optional): Computation mode, either "lhs" (optimized matrix
          summation) or "rhs" (explicit double summation loop). Default: "lhs".
        channel_dropout_p (float, optional): Dropout probability for input
          channels during training to break symmetry. Default: 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        mode: str = "convsum",
        channel_dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode
        self.channel_dropout_p = channel_dropout_p

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_dropout_p > 0.0 and self.training:
            # Generate dropout mask across the channel dimension: (Batch, in_channels, 1, 1)
            mask = torch.bernoulli(
                torch.full(
                    (x.shape[0], self.in_channels, 1, 1),
                    1.0 - self.channel_dropout_p,
                    device=x.device,
                )
            )
            x = x * mask / (1.0 - self.channel_dropout_p)

        if self.mode == "convsum":
            x_sum = x.sum(dim=1, keepdim=True)
            f_sum = self.weight.sum(dim=1, keepdim=True)
            out = F.conv2d(x_sum, f_sum, stride=self.stride, padding=self.padding)

        elif self.mode == "sumconv":
            out = 0
            for i in range(self.in_channels):
                x_i = x[:, i : i + 1, :, :]
                for k in range(self.in_channels):
                    f_ok = self.weight[:, k : k + 1, :, :]
                    out = out + F.conv2d(
                        x_i, f_ok, stride=self.stride, padding=self.padding
                    )
        else:
            raise ValueError(
                f"Unsupported mode: {self.mode}. Choose from 'convsum' or 'sumconv'."
            )

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"mode={self.mode}, "
            f"channel_dropout_p={self.channel_dropout_p}"
        )


class ConvTransposeSum2d(nn.Module):
    """A custom PyTorch 2D transposed convolution module designed to perform operations

    across summed input channels and weights based on the specified mode.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the transposed convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the transposed convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of
          the input. Default: 0.
        output_padding (int or tuple, optional): Additional size added to one side
          of the output shape. Default: 0.
        bias (bool, optional): If True, adds a learnable bias to the output.
          Default: True.
        mode (str, optional): Computation mode, either "convsum" (optimized matrix
          summation) or "sumconv" (explicit double summation loop). Default: "convsum".
        channel_dropout_p (float, optional): Dropout probability for input
          channels during training to break symmetry. Default: 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
        mode: str = "convsum",
        channel_dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.mode = mode
        self.channel_dropout_p = channel_dropout_p

        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_dropout_p > 0.0 and self.training:
            mask = torch.bernoulli(
                torch.full(
                    (x.shape[0], self.in_channels, 1, 1),
                    1.0 - self.channel_dropout_p,
                    device=x.device,
                )
            )
            x = x * mask / (1.0 - self.channel_dropout_p)

        if self.mode == "convsum":
            x_sum = x.sum(dim=1, keepdim=True)
            f_sum = self.weight.sum(dim=0, keepdim=True)
            out = F.conv_transpose2d(
                x_sum,
                f_sum,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
            )

        elif self.mode == "sumconv":
            out = 0
            for i in range(self.in_channels):
                x_i = x[:, i : i + 1, :, :]  # (Batch, 1, H, W)
                f_i = self.weight[i : i + 1, :, :, :].permute(1, 0, 2, 3)

                out = out + F.conv_transpose2d(
                    x_i,
                    f_i,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                )

        else:
            raise ValueError(
                f"Unsupported mode: {self.mode}. Choose from 'convsum' or 'sumconv'."
            )

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"output_padding={self.output_padding}, "
            f"mode={self.mode}, "
            f"channel_dropout_p={self.channel_dropout_p}"
        )
