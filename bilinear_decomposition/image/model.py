import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import einsum, rearrange

# CODE FROM https://github.com/tdooms/bilinear-decomposition

from shared.components import Linear, Bilinear, BilinearConv2d

def _collator(transform=None):
    def inner(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.stack([item[1] for item in batch])
        return (x, y) if transform is None else (transform(x), y)
    return inner

class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        n_layer: int = 1,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        residual: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
    
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        self.residual = residual
        
        
        
        super().__init__(**kwargs)

class Model(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        d_input, d_hidden, d_output = config.d_input, config.d_hidden, config.d_output
        bias, n_layer = config.bias, config.n_layer
        
        self.embed = Linear(d_input, d_hidden, bias=False)
        self.blocks = nn.ModuleList([Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)])
        self.head = Linear(d_hidden, d_output, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()

        def accuracy_fn(y_hat, y):
            if y_hat.dim() == 4:
                preds = y_hat.argmax(dim=1)
                if y.dim() == 1:
                    y = y.view(-1, 1, 1).expand_as(preds)
                return (preds == y).float().mean()
            return (y_hat.argmax(dim=-1) == y).float().mean()

        self.accuracy = accuracy_fn
    
    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x.flatten(start_dim=1))
        
        for layer in self.blocks:
            x = x + layer(x) if self.config.residual else layer(x)
        
        return self.head(x)
    
    @property
    def w_e(self):
        return self.embed.weight.data
    
    @property
    def w_u(self):
        return self.head.weight.data
    
    @property
    def w_lr(self):
        return torch.stack([rearrange(layer.weight.data, "(s o) h -> s o h", s=2) for layer in self.blocks])
    
    @property
    def w_l(self):
        return self.w_lr.unbind(1)[0]
    
    @property
    def w_r(self):
        return self.w_lr.unbind(1)[1]
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return loss, accuracy
    
    def fit(self, train, test, transform=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)
        
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, collate_fn=_collator(transform))
        test_x, test_y = test.x, test.y
        
        pbar = tqdm(range(self.config.epochs))
        history = []
        
        for _ in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }
            
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        
        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

    def decompose(self):
        """The function to decompose a single-layer model into eigenvalues and eigenvectors."""
        
        # Split the bilinear layer into the left and right components
        l, r = self.w_lr[0].unbind()
        
        # Compute the third-order (bilinear) tensor
        b = einsum(self.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
        
        # Symmetrize the tensor
        b = 0.5 * (b + b.mT)

        # Perform the eigendecomposition
        vals, vecs = torch.linalg.eigh(b)
        
        # Project the eigenvectors back to the input space
        vecs = einsum(vecs, self.w_e, "cls emb comp, emb inp -> cls comp inp")
        
        # Return the eigenvalues and eigenvectors
        return vals, vecs


# CODE EXTENSION FOR A BILINEAR CONVOLUTIONAL MODEL

class CNNModel(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)

        d_hidden, d_output = config.d_hidden, config.d_output
        bias, n_layer = config.bias, config.n_layer

        # Fixed kernel size and padding
        in_channels = getattr(config, "in_channels", 1)
        kernel_size = getattr(config, "kernel_size", 14)
        padding = getattr(config, "padding", None)

        self.block = BilinearConv2d(
            in_channels, d_hidden, kernel_size=kernel_size, padding=padding, bias=bias
        )
        
        H_in = config.image_size[0] if hasattr(config, 'image_size') else 28
        W_in = config.image_size[1] if hasattr(config, 'image_size') else 28
        
        H_out = H_in - kernel_size + 2*padding + 1
        W_out = W_in - kernel_size + 2*padding + 1
        
        # Flattened size
        d_flat = d_hidden * H_out * W_out
        
        # Linear head on flattened features
        self.head = nn.Linear(d_flat, d_output, bias=False)

        self.criterion = nn.CrossEntropyLoss()

        def accuracy_fn(y_hat, y):
            if y_hat.dim() == 4:
                preds = y_hat.argmax(dim=1)
                if y.dim() == 1:
                    y = y.view(-1, 1, 1).expand_as(preds)
                return (preds == y).float().mean()
            return (y_hat.argmax(dim=-1) == y).float().mean()

        self.accuracy = accuracy_fn

    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        in_channels = getattr(self.config, "in_channels", 1)
        if x.dim() == 2:
            side = int(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], in_channels, side, side)
 
        x = self.block(x)        # (B, d_hidden, H_out, W_out)
        x = x.flatten(1)         # (B, d_hidden * H_out * W_out)
        return self.head(x)      # (B, d_output)

    @property
    def w_e(self):
        return self.embed.weight.data

    @property
    def w_u(self):
        return self.head.weight.data

    @property
    def w_lr(self):
        return torch.stack([
            rearrange(layer.weight.data, "(s o) c h w -> s o c h w", s=2)
            for layer in self.blocks
        ])

    @property
    def w_l(self):
        return self.w_lr.unbind(1)[0]

    @property
    def w_r(self):
        return self.w_lr.unbind(1)[1]

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new

    def step(self, x, y):
        y_hat = self(x)

        if y_hat.dim() == 4 and y.dim() == 1:
            y = y.view(-1, 1, 1).expand(y.shape[0], y_hat.shape[2], y_hat.shape[3])

        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)

        return loss, accuracy

    def fit(self, train, test, transform=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)

        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, collate_fn=_collator(transform))
        test_x, test_y = test.x, test.y

        pbar = tqdm(range(self.config.epochs))
        history = []

        for _ in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }

            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))

        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

    # Applies the spectral decomposition to the bilinear convolutional model
    def decompose(self, device=None, dtype=None, eig_dtype=None, chunk_c_h: int = 1):
        
        device = torch.device(device) if isinstance(device, str) else (device or self.block.weight.device)
        dtype = dtype or self.block.weight.dtype
        if eig_dtype is None:
            eig_dtype = torch.float32 if device.type == "cpu" and dtype == torch.float16 else dtype

        Q = self._compose_quadratic_form(device=device, dtype=dtype, chunk_c_h=chunk_c_h)

        # Symmetrize
        Q = 0.5 * (Q + Q.mT)
        if Q.dtype != eig_dtype:
            Q = Q.to(eig_dtype)

        # Perform the eigendecomposition
        # eigenvectors have shape (cls, n, n) where each vector has size n = C_in * H_in * W_in
        vals, vecs = torch.linalg.eigh(Q)

        # Reshape eigenvectors back to input space (cls, n, C_in, H_in, W_in)
        # equivalent to the w_e projection in the feedforward case
        # no projection needed since T is already in pixel space
        c_in = self.block.in_channels
        h_in = w_in = int(self.config.d_input ** 0.5)
        vecs = vecs.mT.unflatten(-1, (c_in, h_in, w_in))
        
        return vals, vecs

    # This methods builds the quadratic form directly without materializing the full tensor network since that would take too much time
    def _compose_quadratic_form(self, device=None, dtype=None, chunk_c_h: int = 1) -> Tensor:
        if chunk_c_h < 1:
            raise ValueError("chunk_c_h must be >= 1")

        layer = self.block
        device = torch.device(device) if isinstance(device, str) else (device or layer.weight.device)
        dtype = dtype or layer.weight.dtype

        kernel_height, kernel_width = layer.kernel_size
        stride_height,  stride_width  = layer.stride
        padding_height, padding_width = layer.padding
        dilation_height, dilation_width = layer.dilation

        input_height = input_width = int(self.config.d_input ** 0.5)

        # Extract left/right weight matrices: each (d_hidden, C_in, K_h, K_w)
        parameters_left, parameters_right = rearrange(
            layer.weight.data.to(device=device, dtype=dtype), "(s o) c h w -> s o c h w", s=2
        ).unbind(0)

        # Toeplitz-like matrices encoding the conv geometry (shared for left and right)
        C_w = self._generate_convolution_mechanism_tensor(
            input_width,
            kernel_width,
            stride_width,
            padding_width,
            dilation_width,
            device=device,
            dtype=dtype,
        )
        C_h = self._generate_convolution_mechanism_tensor(
            input_height,
            kernel_height,
            stride_height,
            padding_height,
            dilation_height,
            device=device,
            dtype=dtype,
        )

        output_height = C_h.shape[0]
        output_width = C_w.shape[0]

        d_hidden = parameters_left.shape[0]
        w_u = self.w_u.to(device=device, dtype=dtype)
        expected_flat = d_hidden * output_height * output_width
        if w_u.shape[1] != expected_flat:
            raise ValueError(
                f"head weight expects {expected_flat} features (got {w_u.shape[1]})"
            )

        # Match forward flatten: reshape head weights to (c_h, h_o, w_o).
        head_flat_tn = w_u.unflatten(-1, (d_hidden, output_height, output_width))

        c_in = parameters_left.shape[1]
        n = c_in * input_height * input_width
        classes = w_u.shape[0]
        Q = torch.zeros((classes, n, n), device=device, dtype=dtype)

        for start in range(0, d_hidden, chunk_c_h):
            end = min(start + chunk_c_h, d_hidden)
            params_left = parameters_left[start:end]
            params_right = parameters_right[start:end]

            # Chunk over hidden channels to keep intermediates small.
            left_tn = einsum(
                C_w, C_h, params_left,
                "w_hidden w_in k_w, h_hidden h_in k_h, c_hidden c_in k_h k_w "
                "-> c_in h_in w_in c_hidden h_hidden w_hidden"
            )
            right_tn = einsum(
                C_w, C_h, params_right,
                "w_hidden w_in k_w, h_hidden h_in k_h, c_hidden c_in k_h k_w "
                "-> c_in h_in w_in c_hidden h_hidden w_hidden"
            )

            left_flat = left_tn.flatten(0, 2)
            right_flat = right_tn.flatten(0, 2)
            head_chunk = head_flat_tn[:, start:end]

            Q += einsum(
                left_flat, right_flat, head_chunk,
                "n c_h h_o w_o, m c_h h_o w_o, classes c_h h_o w_o -> classes n m"
            )

        return Q

    # Generates the tensor simulating the convolutional operation mechanism
    def _generate_convolution_mechanism_tensor(self, input_size, kernel_size, stride, padding, dilation):
        output_size = int(((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

        C = torch.zeros((output_size, input_size, kernel_size))

        for output_index in range(output_size):
            for kernel_index in range(kernel_size):
                index = (output_index * stride) + (kernel_index * dilation)
                index = index - padding 

                if 0 <= index < input_size:
                    # This means that in the output_index matrix, input element at position index is multiplied by the kernel element at position kernel_index
                    C[output_index, index, kernel_index] = 1
            
        return C
    
    def _create_spider(self, dimension):
        """Creates a diagonal 3rd-order tensor (spider/copy tensor).

        Entry spider[d, d, d] = 1 for all d, zero elsewhere.
        Used to duplicate inputs (fan-out) and perform element-wise
        multiplication (fan-in) in tensor network notation.
        """
        spider = torch.zeros(dimension, dimension, dimension)
        for d in range(dimension):
            spider[d, d, d] = 1
        return spider
