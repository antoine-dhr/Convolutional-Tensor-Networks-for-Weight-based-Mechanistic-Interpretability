from torch import nn

class BilinearLayer2d(nn.Module):
    def __init__(self, in_channels, upscale_factor, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels * upscale_factor
        
        # Preserves the spatial dimension
        padding = kernel_size // 2

        self.left = nn.Conv2d(in_channels, self.hidden_channels, kernel_size, padding=padding, bias=False)
        self.right = nn.Conv2d(in_channels, self.hidden_channels, kernel_size, padding=padding, bias=False)

        self.projector = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)

        # element-wise multiplication
        combined = x_left * x_right

        return self.projector(combined)
