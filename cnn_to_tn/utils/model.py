import math
from torch import nn
from utils.ResidualBlock import ResidualBlock
from utils.RMSNorm2d import RMSNorm2d

class BaseCNN(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        input_size,
        num_classes
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Number of stages (channels double between stages, spatial dims halve)
        self.num_stages = config.num_stages
        # Number of residual blocks per stage (stacked at same channel width)
        self.blocks_per_stage = config.blocks_per_stage
        # Factor used to upscale the hidden channels in bilinear map
        upscale_factor = config.upscale_factor
        # Channel multiplier between stages
        channels_factor = config.channels_factor
        # Starting output channels
        start_output_channels = config.start_output_channels
        # Kernel size
        kernel_size = config.kernel_size
        # Repeat factor (number of norm+bilinear pairs inside each residual block)
        repeat_factor = config.repeat_factor
        # RMSNorm behavior: channels (default), global, or mixed
        rms_norm_mode = str(getattr(config, "rms_norm_mode", "channels"))

        current_channels = in_channels
        output_channels = start_output_channels
        spatial_dimension = input_size

        for _ in range(self.num_stages):
            for block in range(self.blocks_per_stage):
                if block == 0:
                    # First block of each stage handles the channel change
                    self.layers.append(
                        ResidualBlock(
                            in_channels=current_channels,
                            out_channels=output_channels,
                            upscale_factor=upscale_factor,
                            kernel_size=kernel_size,
                            repeat_factor=repeat_factor,
                            rms_norm_mode=rms_norm_mode,
                        )
                    )
                    current_channels = output_channels
                else:
                    # Remaining blocks keep the same channel width (identity skip)
                    self.layers.append(
                        ResidualBlock(
                            in_channels=current_channels,
                            out_channels=current_channels,
                            upscale_factor=upscale_factor,
                            kernel_size=kernel_size,
                            repeat_factor=repeat_factor,
                            rms_norm_mode=rms_norm_mode,
                        )
                    )

            # Downsample between stages (like ResNet stride-2)
            if spatial_dimension > 2:
                spatial_dimension //= 2
                self.layers.append(nn.AvgPool2d(2))

            # Double channels for the next stage
            output_channels *= channels_factor

        # Global average pooling
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers.append(nn.Flatten(1))
        self.fc = nn.Linear(current_channels, num_classes, bias=False)
        self.layers.append(self.fc)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
