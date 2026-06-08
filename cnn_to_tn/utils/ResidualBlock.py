from torch import nn
from utils.BilinearLayer2d import BilinearLayer2d
from utils.RMSNorm2d import RMSNorm2d

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_factor,
        kernel_size,
        repeat_factor,
        rms_norm_mode="global",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.rms_norm_mode = rms_norm_mode

        current_channels = in_channels

        for _ in range(repeat_factor):
            self.layers.append(
                RMSNorm2d(
                    current_channels,
                    mode=rms_norm_mode,
                )
            )
            self.layers.append(
                BilinearLayer2d(current_channels, upscale_factor, out_channels, kernel_size)
            )
            current_channels = out_channels

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        out = x
        for layer in self.layers:
            out = layer(out)
        # Residual connection
        return identity + out
            
