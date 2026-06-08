from torch import nn
import torch

class RMSNorm2d(nn.Module):
    def __init__(self, normalized_shape, epsilon=0, mode="global"):
        super().__init__()
        self.epsilon = epsilon 
        self.mode = mode
        self.gamma = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))

    def forward(self, x):
        # x has shape (B, C, H, W)
        if self.mode == "channels":
            mean_squared = x.pow(2).mean(dim=1, keepdim=True)
            x = x * torch.rsqrt(mean_squared + self.epsilon)
        elif self.mode == "global":
            mean_squared = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
            x = x * torch.rsqrt(mean_squared + self.epsilon)
        else:
            raise ValueError(f"Unsupported RMSNorm2d mode: {self.mode}")

        # (x_i / RMS(x)) * gamma (or (x / constant) * gamma for learned_constant mode)
        return x * self.gamma
