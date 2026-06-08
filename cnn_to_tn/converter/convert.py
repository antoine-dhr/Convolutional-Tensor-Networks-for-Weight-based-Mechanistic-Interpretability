import torch
from torch.nn import Conv2d, AvgPool2d, Linear, AdaptiveAvgPool2d, Flatten, Identity
import quimb.tensor as qtn

from utils.BilinearLayer2d import BilinearLayer2d
from utils.RMSNorm2d import RMSNorm2d
from utils.ResidualBlock import ResidualBlock
from utils.model import BaseCNN

class Convert:
    def __init__(self, input_size):

        super().__init__()

        self.current_size = input_size 

    def convert_model(self, model: BaseCNN):
        # The resulting tensor network
        tn_current = None
        # Stage counter
        stage_id = 0
        # Residual block counter
        residual_block_count = 1

        number_of_residual_blocks = model.num_stages * model.blocks_per_stage

        # Iterates over every layer of the model, and for each layer, convert it to its equivalent tensor network
        # The resulting tensor network component then needs to be chained to the global resulting tensor network
        for layer in model.layers:
            if isinstance(layer, ResidualBlock):
                # Last residual block must have double_copy = False
                block_tn = self.convert_residual_block(layer, double_copy= not (residual_block_count == number_of_residual_blocks))
                if tn_current is None:
                    tn_current = block_tn
                else:
                    tn_current = self._attach_block_stage(tn_current, block_tn, stage_id)
                stage_id += 1
                residual_block_count += 1

            elif isinstance(layer, AvgPool2d):
                pool_tn = self.convert_avgpool2d(layer)
                if tn_current is None:
                    tn_current = pool_tn
                else:
                    tn_current = self._attach_linear_stage(tn_current, pool_tn, stage_id)
                stage_id += 1

            elif isinstance(layer, AdaptiveAvgPool2d):
                gap_tn = self.convert_adaptive_avg_pool_2d(layer)
                if tn_current is None:
                    tn_current = gap_tn
                else:
                    tn_current = self._attach_linear_stage(tn_current, gap_tn, stage_id)
                stage_id += 1

            elif isinstance(layer, Flatten):
                flatten_t = self.convert_flatten()
                tn_current = self._attach_flatten_stage(tn_current, flatten_t, stage_id)
                stage_id += 1

            elif isinstance(layer, Linear):
                linear_t = self.convert_linear(layer)
                tn_current = self._attach_linear_tensor(tn_current, linear_t, stage_id)
                stage_id += 1

            else:
                raise AssertionError(
                    f"Unsupported layer in model conversion: {type(layer)}"
                )

        return tn_current

    # Double copy indicates whether the residual block must keep a copy of the data channels in the skip channels too
    # This is needed when the residual block is not the last one of the entire architecture
    def convert_residual_block(self, residual_block: ResidualBlock, double_copy: bool = True):
        total_in, H, W = self.current_size
        assert (total_in - 2) % 2 == 0, (
            "Residual block input must be (1, RMS-TERM-PREV, data, data)"
        )
        data_channels = (total_in - 2) // 2

        rms_layers = []
        bilinear_layers = []
        for layer in residual_block.layers:
            if isinstance(layer, RMSNorm2d):
                rms_layers.append(layer)
            elif isinstance(layer, BilinearLayer2d):
                bilinear_layers.append(layer)
            else:
                raise AssertionError(
                    f"Unsupported layer in ResidualBlock conversion: {type(layer)}"
                )

        assert len(rms_layers) == len(bilinear_layers), (
            "ResidualBlock must alternate RMSNorm2d and BilinearLayer2d"
        )
        assert len(rms_layers) > 0, "ResidualBlock has no RMS/Bilinear pairs"

        like = self._infer_like_from_module(residual_block)

        # Step 1: convert the skip connection
        if isinstance(residual_block.skip, Identity):
            tn_current = self._identity_linear(like)
            copy_input_size = data_channels
        elif isinstance(residual_block.skip, Conv2d):
            meta_channels = 2 + data_channels
            tn_current = self.convert_conv2d_extended(
                conv_layer=residual_block.skip,
                data_start=meta_channels,
                data_len=total_in - meta_channels,
                unique_label="skip",
                to_hidden=False,
            )
            tn_current = tn_current.reindex({
                "C_combined": "C_in",
                "H_combined": "H_in",
                "W_combined": "W_in",
            })
            copy_input_size = residual_block.skip.out_channels
        else:
            raise AssertionError(
                f"Unsupported skip connection in ResidualBlock: {type(residual_block.skip)}"
            )

        stage_id = 0

        # Step 2: convert the chain of RMS layers and bilinear layers to an equivalent tensor network
        for idx, (rms_layer, bilinear_layer) in enumerate(zip(rms_layers, bilinear_layers)):
            is_last = idx == len(rms_layers) - 1

            assert bilinear_layer.in_channels == data_channels, (
                "Bilinear input channels must match current data block size"
            )

            # If this is the last RMS layer, we need to compute the RMS of the current input we have in the data channels
            # See the thesis why; look at the computational simplifications of the architecture
            if is_last:
                rms_tn = self.convert_rms(
                    rms_layer=rms_layer,
                    bilinear_size=data_channels,
                    copy_input_size=copy_input_size,
                )
            # If this is not the last RMS layer, we only need to apply the gamma scaling
            else:
                rms_tn = self.convert_gamma(
                    rms_layer=rms_layer,
                    bilinear_size=data_channels,
                    copy_input_size=copy_input_size,
                )

            # We attach the result to the current tensor network we are building; we chain the results together
            tn_current = self._attach_bilinear_stage(tn_current, rms_tn, stage_id)
            stage_id += 1

            rms_count = 2 if is_last else 1
            # Convert the bilinear layer
            bilinear_tn = self.convert_bilinear_layer(
                bilinear_layer=bilinear_layer,
                copy_input_size=copy_input_size,
                rms_count=rms_count,
            )
            # Chain the bilinear layer to the current tensor network we are building
            tn_current = self._attach_bilinear_stage(tn_current, bilinear_tn, stage_id)
            stage_id += 1

            data_channels = bilinear_layer.out_channels

        assert data_channels == copy_input_size, (
            "Residual connection requires skip and data channels to match"
        )

        # Step 3: we have computed the skip connection and the chain of alternating RMS and bilinear layer
        # We now need to combine both and implement the residual connection; f(x) + skip(x)
        # This generates a new tensor network component
        residual_tn = self.convert_residual_connection(
            block_size=data_channels,
            double_copy=double_copy,
        )
        # We attach the tensor network component to the final result
        tn_current = self._attach_bilinear_stage(tn_current, residual_tn, stage_id)

        return tn_current
    
        # Help function to create a spider
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

    def _infer_like_from_module(self, module):
        for param in module.parameters():
            return dict(dtype=param.dtype, device=param.device)
        return dict(dtype=torch.get_default_dtype(), device=torch.device("cpu"))

    def _identity_linear(self, like):
        channels, height, width = self.current_size
        eye_c = torch.eye(channels, **like)
        eye_h = torch.eye(height, **like)
        eye_w = torch.eye(width, **like)
        return qtn.TensorNetwork([
            qtn.Tensor(eye_c, inds=("C_out", "C_in")),
            qtn.Tensor(eye_h, inds=("H_out", "H_in")),
            qtn.Tensor(eye_w, inds=("W_out", "W_in")),
        ])

    # Since we build the resulting tensor network piece by piece by converting each layer of the model to an equivalent tensor network component,
    # we need a mechanism to chain all components together and ensure they all have unique indices 
    # This is important because if they have the same indices, then we are contracting over them, and that is not something we want
    # So the following series of functions help in this chaining of results and ensuring that indices are unique

    # Renames all indices of a tensor network given a suffix
    def _reindex_all_with_suffix(self, network, suffix):
        mapping = {ind: f"{ind}{suffix}" for ind in network.ind_map}
        return network.reindex(mapping)

    def _prepare_bilinear_stage(self, stage_tn, suffix):
        staged = self._reindex_all_with_suffix(stage_tn, suffix)
        return staged.reindex({
            f"C_in_L{suffix}": "C_in_L",
            f"H_in_L{suffix}": "H_in_L",
            f"W_in_L{suffix}": "W_in_L",
            f"C_in_R{suffix}": "C_in_R",
            f"H_in_R{suffix}": "H_in_R",
            f"W_in_R{suffix}": "W_in_R",
            f"C_out{suffix}": "C_out",
            f"H_out{suffix}": "H_out",
            f"W_out{suffix}": "W_out",
        })

    # This method chains stage_tn to prev_tn
    # This connection is bilinear, meaning that stage_tn expects two copies of prev_tn
    # This also means that all indices must be renamed in prev_tn to ensure that they are unique and we do not mix indices from the left and right side of the stage_tn input branches
    def _attach_bilinear_stage(self, prev_tn, stage_tn, stage_id):
        stage_suffix = f"_s{stage_id}"
        stage = self._prepare_bilinear_stage(stage_tn, stage_suffix)

        left_suffix = f"_L{stage_id}"
        right_suffix = f"_R{stage_id}"

        # Reindex left and right branches
        left = self._reindex_all_with_suffix(prev_tn, left_suffix)
        right = self._reindex_all_with_suffix(prev_tn, right_suffix)

        left = left.reindex({
            f"C_out{left_suffix}": "C_in_L",
            f"H_out{left_suffix}": "H_in_L",
            f"W_out{left_suffix}": "W_in_L",
        })
        right = right.reindex({
            f"C_out{right_suffix}": "C_in_R",
            f"H_out{right_suffix}": "H_in_R",
            f"W_out{right_suffix}": "W_in_R",
        })

        # Duplicate the previous network for left/right inputs to preserve bilinear structure.
        return left & right & stage

    # This methods chains a residual block tensor network (block_tn) to the current tensor network we are building (prev_tn)
    def _attach_block_stage(self, prev_tn, block_tn, stage_id):
        stage_suffix = f"_s{stage_id}"
        stage = self._reindex_all_with_suffix(block_tn, stage_suffix)

        outer = set(stage.outer_inds())
        c_inds = sorted(i for i in outer if i.startswith("C_in"))

        stage = stage.reindex({
            f"C_out{stage_suffix}": "C_out",
            f"H_out{stage_suffix}": "H_out",
            f"W_out{stage_suffix}": "W_out",
        })

        result = stage
        for slot_idx, c_ind in enumerate(c_inds):
            slot_local = c_ind[len("C_in") :]
            h_ind = f"H_in{slot_local}"
            w_ind = f"W_in{slot_local}"

            slot_tag = f"_b{stage_id}p{slot_idx}"
            prev_copy = self._reindex_all_with_suffix(prev_tn, slot_tag)
            prev_copy = prev_copy.reindex({
                f"C_out{slot_tag}": c_ind,
                f"H_out{slot_tag}": h_ind,
                f"W_out{slot_tag}": w_ind,
            })
            result = result & prev_copy

        return result

    # This method chains average pooling and global average pooling tensor networks to the tensor network we are currently building
    def _attach_linear_stage(self, prev_tn, stage_tn, stage_id):
        stage_suffix = f"_s{stage_id}"
        stage = self._reindex_all_with_suffix(stage_tn, stage_suffix)

        if f"C_combined{stage_suffix}" in stage.ind_map:
            c_in = f"C_combined{stage_suffix}"
            h_in = f"H_combined{stage_suffix}"
            w_in = f"W_combined{stage_suffix}"
        elif f"C_in{stage_suffix}" in stage.ind_map:
            c_in = f"C_in{stage_suffix}"
            h_in = f"H_in{stage_suffix}"
            w_in = f"W_in{stage_suffix}"
        else:
            raise AssertionError("Unsupported linear stage input indices")

        c_out = f"C_out{stage_suffix}"
        h_out = f"H_out{stage_suffix}"
        w_out = f"W_out{stage_suffix}"

        c_mid = f"C_mid{stage_suffix}"
        h_mid = f"H_mid{stage_suffix}"
        w_mid = f"W_mid{stage_suffix}"

        prev = prev_tn.reindex({
            "C_out": c_mid,
            "H_out": h_mid,
            "W_out": w_mid,
        })
        stage = stage.reindex({
            c_in: c_mid,
            h_in: h_mid,
            w_in: w_mid,
            c_out: "C_out",
            h_out: "H_out",
            w_out: "W_out",
        })

        return prev & stage

    # This method chains the flatten tensor network to the current tensor network we are building
    def _attach_flatten_stage(self, prev_tn, flatten_tn, stage_id):

        stage_suffix = f"_s{stage_id}"
        stage = flatten_tn.reindex({
            "C_in": f"C_in{stage_suffix}",
            "H_in": f"H_in{stage_suffix}",
            "W_in": f"W_in{stage_suffix}",
            "flatten": f"flatten{stage_suffix}",
        })
        prev = prev_tn.reindex({
            "C_out": f"C_in{stage_suffix}",
            "H_out": f"H_in{stage_suffix}",
            "W_out": f"W_in{stage_suffix}",
        })
        stage = stage.reindex({f"flatten{stage_suffix}": "flatten"})

        return prev & stage

    # This method chains a matrix (linear_tn) to the final tensor network we are building
    def _attach_linear_tensor(self, prev_tn, linear_tn, stage_id):

        stage_suffix = f"_s{stage_id}"
        stage = linear_tn.reindex({
            "flatten": f"flatten{stage_suffix}",
            "linear_out": f"linear_out{stage_suffix}",
        })
        prev = prev_tn.reindex({"flatten": f"flatten{stage_suffix}"})
        stage = stage.reindex({f"linear_out{stage_suffix}": "linear_out"})

        return prev & stage

    # This method converts a convolutional layer to its equivalent tensor network component
    # The resulting tensor network simulates the convolution on some data channels, and leaves the other channels untouched
    def convert_conv2d_extended(self, conv_layer, data_start, data_len, unique_label='', to_hidden=False):
        """
        Tensor network conversion implementing
            (prefix | data | suffix) --> (prefix | Conv(data) | suffix)

        data_start: start index of the data slice
        data_len:   length of the data slice (must match conv_layer.in_channels)
        """
        total_channels, H_in, W_in = self.current_size
        assert 0 <= data_start <= total_channels, "data_start out of range"
        assert data_len > 0, "data_len must be > 0"
        assert data_start + data_len <= total_channels, "data slice exceeds channel dim"
        assert data_len == conv_layer.in_channels, (
            "data_len must match conv_layer.in_channels"
        )

        prefix = data_start
        suffix = total_channels - data_start - data_len

        # Tensor network for the data channels only, no pass through mechanism yet
        C_w_data, C_h_data, params_data = self._convert_conv2d(
            (data_len, H_in, W_in), conv_layer)

        C_out = conv_layer.out_channels
        k_h, k_w = conv_layer.kernel_size
        H_out = C_h_data.shape[0]
        W_out = C_w_data.shape[0]

        assert H_out == H_in, (
            f"Spatial height must be preserved for bypass: H_in={H_in}, H_out={H_out}. "
            f"Use same-spatial-dimension padding (e.g. padding=(k_h-1)//2, stride=1)."
        )
        assert W_out == W_in, (
            f"Spatial width must be preserved for bypass: W_in={W_in}, W_out={W_out}. "
            f"Use same-spatial-dimension padding (e.g. padding=(k_w-1)//2, stride=1)."
        )

        # The current tensor network we have (C_w_data, C_h_data, params_data) works for the data channels only, it does not work for inputs of the form (1, Lambda, x, y)
        # We need to extend those to add a pass through mechanism
        like = dict(dtype=conv_layer.weight.dtype, device=conv_layer.weight.device)

        C_h_ext = torch.zeros(H_out, H_in, k_h + 1, **like)
        C_h_ext[:, :, :k_h] = C_h_data.to(**like)
        C_h_ext[:, :, k_h] = torch.eye(H_in, **like)

        C_w_ext = torch.zeros(W_out, W_in, k_w + 1, **like)
        C_w_ext[:, :, :k_w] = C_w_data.to(**like)
        C_w_ext[:, :, k_w] = torch.eye(W_in, **like)

        total_out = prefix + C_out + suffix
        params_ext = torch.zeros(total_out, total_channels, k_h + 1, k_w + 1, **like)

        # Applies the convolution to the data channels
        params_ext[
            data_start:data_start + C_out,
            data_start:data_start + data_len,
            :k_h,
            :k_w,
        ] = params_data.to(**like)

        # Implements the pass through mechanism for the prefix channels
        for c in range(prefix):
            params_ext[c, c, k_h, k_w] = 1.0

        # Implements the pass through mechanism for the suffix channels
        for s in range(suffix):
            in_c = data_start + data_len + s
            out_c = data_start + C_out + s
            params_ext[out_c, in_c, k_h, k_w] = 1.0

        self.current_size = (total_out, H_out, W_out)

        output_label = ('_hidden_' + unique_label) if to_hidden else '_out'
        input_label = ('_in_' + unique_label) if to_hidden else '_combined'

        C_w_tn = qtn.Tensor(
            data=C_w_ext,
            inds=(f'W{output_label}', f'W{input_label}', f'K_w_{unique_label}'),
            tags={f'C_w_{unique_label}'},
        )
        C_h_tn = qtn.Tensor(
            data=C_h_ext,
            inds=(f'H{output_label}', f'H{input_label}', f'K_h_{unique_label}'),
            tags={f'C_h_{unique_label}'},
        )
        parameters_t = qtn.Tensor(
            data=params_ext,
            inds=(f'C{output_label}', f'C{input_label}', f'K_h_{unique_label}', f'K_w_{unique_label}'),
            tags={f'K_{unique_label}'},
        )

        return qtn.TensorNetwork([C_w_tn, C_h_tn, parameters_t])
    
    # Converts a convolutional layer to an equivalent tensor network that is only concerned with simulating the convolutional layer on some input data
    # It does not account for the pass through mechanism or inputs of the form (1, Lambda, x, y)
    # Now it only works on inputs of the form x
    def _convert_conv2d(self, input_size: tuple[int, int, int], layer: Conv2d):

        # Current implementation supports no bias
        assert layer.bias == None, "Conv2d layer must have bias=False for tensor network conversion"

        _, input_height, input_width = input_size

        kernel_height, kernel_width = layer.kernel_size
        stride_height, stride_width = layer.stride 
        padding_height, padding_width = layer.padding
        dilation_height, dilation_width = layer.dilation 

        output_height = ((input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height) + 1
        output_width =  ((input_width  + 2 * padding_width  - dilation_width  * (kernel_width - 1)  - 1) // stride_width)  + 1
        output_channels = layer.out_channels

        output_size = (output_channels, output_height, output_width)
        self.current_size = output_size

        # Third-order tensor applying the convolutional mechanism to the width of the feature map
        C_w = self._generate_convolution_mechanism_tensor(input_width, kernel_width, stride_width, padding_width, dilation_width) 
        # Third-order tensor applying the convolutional mechanism to the height of the feature map
        C_h = self._generate_convolution_mechanism_tensor(input_height, kernel_height, stride_height, padding_height, dilation_height) 

        with torch.no_grad():
            # Fourth-order tensor containing the parameters of the kernel
            parameters = layer.weight.detach()

        return C_w, C_h, parameters

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
    
    # This method converts the LAST RMS layer of a residual block to its equivalent tensor network
    def convert_rms(self, rms_layer, bilinear_size, copy_input_size, unique_label=''):
        """
        Tensor network conversion implementing
            (1, RMS-TERM-PREV, bilinear_output, copy_input) --> (1, RMS-TERM-NEW, RMS-TERM-PREV, gamma * bilinear_output, copy_input)
        with
            RMS-TERM-NEW = RMS(bilinear_output)^2 = (bilinear_output ** 2).mean()
        """

        assert rms_layer.mode == "global", "Conversion only works for global mode"

        total_in, H, W = self.current_size
        expected_in = 1 + 1 + bilinear_size + copy_input_size
        assert total_in == expected_in, (
            f"current_size channel dim must be 1 (constant channel) + 1 (RMS layer) + bilinear_output_size + copy_input_size = {expected_in}; "
            f"got {total_in} channels"
        )

        gamma = rms_layer.gamma.detach().reshape(-1)

        like = dict(dtype=gamma.dtype, device=gamma.device)

        # 1 for the constant-1 channel
        # 1 for the new RMS-term we introduce
        # 1 for the previous RMS-term
        # bilinear size for the bilinear output we keep
        # copy input size for the size of the input we copy over the channels to add to the residual at the end
        out_C = 1 + 1 + 1 + bilinear_size + copy_input_size

        # The bilinear channels start after the first two channels; the constant channel and the RMS channel
        bilinear_start = 1 + 1
        bilinear_stop = bilinear_start + bilinear_size

        # The gating mechanism allows the tensor K to simulate two operations at the same time
        # One gate will implement the pass through mechanism
        # The other gate implements the RMS computation
        gate_pass = 0
        gate_rms = 1

        # K[c_out, c_L, c_R, gate]
        K = torch.zeros(out_C, total_in, total_in, 2, **like)

        # Gate 0: pass through mechanism
        K[0, 0, 0, gate_pass] = 1.0

        # Passthrough for RMS-TERM-PREV, bilinear_output and copy_input
        # Bilinear channels are scaled by gamma
        for c_in in range(1, total_in):
            c_out = c_in + 1
            scale = 1.0
            if bilinear_start <= c_in < bilinear_stop:
                scale = gamma[c_in - bilinear_start]
            K[c_out, c_in, 0, gate_pass] = scale

        # Gate 1: RMS(B)^2 channel from raw bilinear channels (before gamma), averaged over channels.
        rms_channel_scale = 1.0 / float(bilinear_size)
        for c in range(bilinear_start, bilinear_stop):
            K[1, c, c, gate_rms] = rms_channel_scale

        H_t_data = torch.zeros(H, H, H, 2, **like)
        W_t_data = torch.zeros(W, W, W, 2, **like)

        # Gate 0: spatial passthrough (local)
        H_t_data[:, :, :, gate_pass] = self._create_spider(H).to(**like)
        W_t_data[:, :, :, gate_pass] = self._create_spider(W).to(**like)

        # Gate 1: spatial global averaging with broadcast
        # For each output (h_out, w_out), compute mean over input positions while enforcing h_L == h_R and w_L == w_R (square at same spatial location)
        h_diag = torch.arange(H, device=like['device'])
        w_diag = torch.arange(W, device=like['device'])
        H_t_data[:, h_diag, h_diag, gate_rms] = 1.0 / float(H)
        W_t_data[:, w_diag, w_diag, gate_rms] = 1.0 / float(W)

        suf = f'_{unique_label}' if unique_label else ''
        C_out_idx = f'C_out{suf}'
        H_out_idx = f'H_out{suf}'
        W_out_idx = f'W_out{suf}'
        C_in_L_idx = f'C_in_L{suf}'
        H_in_L_idx = f'H_in_L{suf}'
        W_in_L_idx = f'W_in_L{suf}'
        C_in_R_idx = f'C_in_R{suf}'
        H_in_R_idx = f'H_in_R{suf}'
        W_in_R_idx = f'W_in_R{suf}'
        G_idx = f'G_rms{suf}'

        K_tn = qtn.Tensor(
            data=K,
            inds=(C_out_idx, C_in_L_idx, C_in_R_idx, G_idx),
            tags={'K_rms'},
        )
        H_tn = qtn.Tensor(
            data=H_t_data,
            inds=(H_out_idx, H_in_L_idx, H_in_R_idx, G_idx),
            tags={'S_H_rms'},
        )
        W_tn = qtn.Tensor(
            data=W_t_data,
            inds=(W_out_idx, W_in_L_idx, W_in_R_idx, G_idx),
            tags={'S_W_rms'},
        )

        self.current_size = (out_C, H, W)

        return qtn.TensorNetwork([K_tn, H_tn, W_tn])
    
    # This methods converts all RMS layers EXCEPT the last one of a residual block
    # It implements the gamma scaling
    def convert_gamma(self, rms_layer, bilinear_size, copy_input_size, unique_label=''):
        """
        Tensor network conversion implementing
            (1, RMS-TERM, bilinear_output, copy_input) -> (1, RMS-TERM, gamma * bilinear_output, copy_input)
        """
        total_in, H, W = self.current_size
        expected_in = 1 + 1 + bilinear_size + copy_input_size
        assert total_in == expected_in, (
            f"current_size channel dim must be 1 + 1 + bilinear_size + copy_input_size = {expected_in}; "
            f"got {total_in}"
        )

        gamma = rms_layer.gamma.detach().reshape(-1)
        assert gamma.numel() == bilinear_size, (
            f"rms_layer.gamma must have one value per bilinear channel: "
            f"expected {bilinear_size}, got {gamma.numel()}"
        )
        like = dict(dtype=gamma.dtype, device=gamma.device)

        bilinear_start = 2 
        bilinear_stop = bilinear_start + bilinear_size

        # K is responsible to implement the gamma scaling and pass through mechanism
        K = torch.zeros(total_in, total_in, total_in, **like)
        for c in range(total_in):
            scale = gamma[c - bilinear_start] if bilinear_start <= c < bilinear_stop else 1.0
            K[c, c, 0] = scale

        suf = f'_{unique_label}' if unique_label else ''
        K_tn = qtn.Tensor(
            data=K,
            inds=(f'C_out{suf}', f'C_in_L{suf}', f'C_in_R{suf}'),
            tags={'K_gamma'},
        )
        H_tn = qtn.Tensor(
            data=self._create_spider(H).to(**like),
            inds=(f'H_out{suf}', f'H_in_L{suf}', f'H_in_R{suf}'),
            tags={'S_H_gamma'},
        )
        W_tn = qtn.Tensor(
            data=self._create_spider(W).to(**like),
            inds=(f'W_out{suf}', f'W_in_L{suf}', f'W_in_R{suf}'),
            tags={'S_W_gamma'},
        )

        self.current_size = (total_in, H, W)

        return qtn.TensorNetwork([K_tn, H_tn, W_tn])
    
    # This method converts a bilinear layer to an equivalent tensor network component
    # The bilinear layer conversion has two cases;
    # One case is when the input has 1 RMS term (always except after the last RMS layer of a residual block)
    # The second case is when the input has 2 RMS terms (only after the last RMS layer of a residual block)
    def convert_bilinear_layer(self, bilinear_layer, copy_input_size, unique_label='', rms_count=1):
        """
        Builds a tensor network implementing:
            (1, RMS-term, bilinear_output, copy_input) -> (1, RMS-term, bilinear_layer(bilinear_output), copy_input)

            rms_count=1: (1, RMS-term, bilinear_output, copy_input) -> (1, RMS-term, bilinear_layer(bilinear_output), copy_input)
            rms_count=2: (1, RMS-term-1, RMS-term-2, b_o, copy_input) -> (1, RMS-term-1, RMS-term-2, bilinear_layer(bilinear_output), copy_input)
        """

        assert isinstance(bilinear_layer, BilinearLayer2d), "bilinear_layer must be a BilinearLayer2d"

        total_in, H, W = self.current_size
        bilinear_block_size = bilinear_layer.in_channels
        bilinear_out_channels = bilinear_layer.out_channels
        hidden_channels = bilinear_layer.hidden_channels

        # Prefix: constant + RMS-term
        # Data: bilinear block
        # Suffix: copy of input for residual add
        prefix_channels = 1 + rms_count
        suffix_channels = copy_input_size
        meta_channels = prefix_channels + suffix_channels

        # The expected number of input channels
        expected_in = 1 + rms_count + bilinear_block_size + copy_input_size

        assert total_in == expected_in, (
            f"current_size channel dim must be 1 (constant) + {rms_count} (RMS-term(s)) + bilinear input size + copy input size = {expected_in}; "
            f"got {total_in} channels"
        )

        like = dict(
            dtype=bilinear_layer.left.weight.dtype,
            device=bilinear_layer.left.weight.device,
        )

        data_start = prefix_channels

        # Step 1: convert the first two convolutional layers of the bilinear layer equation
        # Left and right: (prefix | b_o | suffix) -> (prefix | hidden(b_o) | suffix)
        left_tn = self.convert_conv2d_extended(
            conv_layer=bilinear_layer.left,
            data_start=data_start,
            data_len=bilinear_block_size,
            unique_label='L',
            to_hidden=True,
        )

        # Reset because convert_conv2d_extended updates current_size
        self.current_size = (total_in, H, W)

        right_tn = self.convert_conv2d_extended(
            conv_layer=bilinear_layer.right,
            data_start=data_start,
            data_len=bilinear_block_size,
            unique_label='R',
            to_hidden=True,
        )

        # Stpe 2: combine all components with each other
        combine_channels = torch.zeros(
            meta_channels + hidden_channels,
            meta_channels + hidden_channels,
            meta_channels + hidden_channels,
            **like,
        )
        # Pass through mechanism
        for m in range(prefix_channels):
            combine_channels[m, m, 0] = 1.0
        for h in range(hidden_channels):
            idx = prefix_channels + h
            combine_channels[idx, idx, idx] = 1.0
        for s in range(suffix_channels):
            idx = prefix_channels + hidden_channels + s
            combine_channels[idx, idx, 0] = 1.0

        combine_channels_tn = qtn.Tensor(
            data=combine_channels,
            inds=('C_combined', 'C_hidden_L', 'C_hidden_R'),
            tags={'S_C_bilinear'},
        )
        combine_height_tn = qtn.Tensor(
            data=self._create_spider(H).to(**like),
            inds=('H_combined', 'H_hidden_L', 'H_hidden_R'),
            tags={'S_H_bilinear'},
        )
        combine_width_tn = qtn.Tensor(
            data=self._create_spider(W).to(**like),
            inds=('W_combined', 'W_hidden_L', 'W_hidden_R'),
            tags={'S_W_bilinear'},
        )

        # Step 3: convert the last convolutional layer of the bilinear layer equation
        # Projector: (prefix | hidden | suffix) -> (prefix | conv(hidden) | suffix) = (prefix | bilinear(input) | suffix)
        self.current_size = (meta_channels + hidden_channels, H, W)
        projection_tn = self.convert_conv2d_extended(
            conv_layer=bilinear_layer.projector,
            data_start=prefix_channels,
            data_len=hidden_channels,
            unique_label='',
            to_hidden=False,
        )

        out_channels = 1 + rms_count + bilinear_out_channels + copy_input_size

        suf = f'_{unique_label}' if unique_label else ''
        C_in_L_idx = f'C_in_L{suf}'
        H_in_L_idx = f'H_in_L{suf}'
        W_in_L_idx = f'W_in_L{suf}'
        C_in_R_idx = f'C_in_R{suf}'
        H_in_R_idx = f'H_in_R{suf}'
        W_in_R_idx = f'W_in_R{suf}'
        C_out_idx = f'C_out{suf}'
        H_out_idx = f'H_out{suf}'
        W_out_idx = f'W_out{suf}'

        left_tn = left_tn.reindex({
            'C_in_L': C_in_L_idx,
            'H_in_L': H_in_L_idx,
            'W_in_L': W_in_L_idx,
        })
        right_tn = right_tn.reindex({
            'C_in_R': C_in_R_idx,
            'H_in_R': H_in_R_idx,
            'W_in_R': W_in_R_idx,
        })
        projection_tn = projection_tn.reindex({
            'C_out': C_out_idx,
            'H_out': H_out_idx,
            'W_out': W_out_idx,
        })

        self.current_size = (out_channels, H, W)

        return (
            left_tn
            & right_tn
            & projection_tn
            & qtn.TensorNetwork([
                combine_channels_tn,
                combine_height_tn,
                combine_width_tn,
            ])
        )
    
    # This method converts the residual connection mechanism to a tensor network component
    # The residual block mechanism is responsible for combining the necessary components from the input to form the output of the residual block
    # The double_copy flag determines whether we need a copy for the next skip data or not, this is always needed except for the last residual block
    def convert_residual_connection(self, block_size, unique_label='', double_copy=False):
        """
        Builds a tensor network implementing:
        (1, RMS-TERM-1, RMS-TERM-2, bilinear_output(n), copy_input(n)) -> (1, RMS-TERM-1*RMS-TERM-2, RMS-TERM-2*bilinear_output + RMS-TERM-1*copy_input [, copy])
        """
        total_in, H, W = self.current_size
        n = block_size
        out_C = 2 + (2 if double_copy else 1) * n

        K = torch.zeros(out_C, total_in, total_in)
        # Constant pass through
        K[0, 0, 0] = 1.0      
        # RMS_1(L) * RMS_2(R)
        K[1, 1, 2] = 1.0 
        for c in range(n):
            bilinear_ch = 3 + c
            copy_ch     = n + 3 + c
            out_ch      = 2 + c
            # RMS_2(L) * bilinear(R)
            K[out_ch, 2, bilinear_ch] = 1.0 
            # RMS_1(L) * copy(R)
            K[out_ch, 1, copy_ch]     = 1.0
            if double_copy:
                second_out_ch = n + 2 + c
                K[second_out_ch, 2, bilinear_ch] = 1.0
                K[second_out_ch, 1, copy_ch]     = 1.0

        suf = f'_{unique_label}' if unique_label else ''
        K_tn = qtn.Tensor(
            data=K,
            inds=(f'C_out{suf}', f'C_in_L{suf}', f'C_in_R{suf}'),
            tags={'K_residual_rms'},
        )
        H_tn = qtn.Tensor(
            data=self._create_spider(H),
            inds=(f'H_out{suf}', f'H_in_L{suf}', f'H_in_R{suf}'),
            tags={'S_H_residual_rms'},
        )
        W_tn = qtn.Tensor(
            data=self._create_spider(W),
            inds=(f'W_out{suf}', f'W_in_L{suf}', f'W_in_R{suf}'),
            tags={'S_W_residual_rms'},
        )

        self.current_size = (out_C, H, W)

        return qtn.TensorNetwork([K_tn, H_tn, W_tn])


    # This function converts an average pooling layer to its equivalent tensor network
    # This is relatively easy since we already have the mechanism to do that in the convolutional layer conversion
    # Since an average pooling layer is just a special case of a convolutional layer, we can reuse that code and modify its output slightly
    def convert_avgpool2d(self, pool_layer: AvgPool2d, unique_label: str = ''):
        """
        Builds a tensor network implementing:
            (1, RMS-TERM, data, data) -> (1, RMS-TERM, AVGPool(data), AVGPool(data))
        """
        def _pair(x):
            if isinstance(x, tuple):
                return x
            return (x, x)

        total_channels, H_in, W_in = self.current_size

        kernel_height, kernel_width = _pair(pool_layer.kernel_size)
        stride_height, stride_width = _pair(
            pool_layer.stride if pool_layer.stride is not None else pool_layer.kernel_size
        )
        padding_height, padding_width = _pair(pool_layer.padding)
        dilation_height, dilation_width = (1, 1)

        H_out = ((H_in + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height) + 1
        W_out = ((W_in + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width) + 1

        self.current_size = (total_channels, H_out, W_out)

        C_w = self._generate_convolution_mechanism_tensor(W_in, kernel_width, stride_width, padding_width, dilation_width)
        C_h = self._generate_convolution_mechanism_tensor(H_in, kernel_height, stride_height, padding_height, dilation_height)

        with torch.no_grad():
            parameters = torch.zeros(
                (total_channels, total_channels, kernel_height, kernel_width),
                dtype=torch.get_default_dtype(),
            )
            scale = 1.0 / float(kernel_height * kernel_width)
            diag = torch.arange(total_channels)
            parameters[diag, diag, :, :] = scale

        s = f'_{unique_label}' if unique_label else ''
        in_label = f'_in{s}' if unique_label else '_combined'
        out_label = f'_out{s}' if unique_label else '_out'

        C_w_tn = qtn.Tensor(
            data=C_w,
            inds=(f'W{out_label}', f'W{in_label}', f'K_w_pool{s}'),
            tags={f'C_w_pool{s}'}
        )
        C_h_tn = qtn.Tensor(
            data=C_h,
            inds=(f'H{out_label}', f'H{in_label}', f'K_h_pool{s}'),
            tags={f'C_h_pool{s}'}
        )
        parameters_t = qtn.Tensor(
            data=parameters,
            inds=(f'C{out_label}', f'C{in_label}', f'K_h_pool{s}', f'K_w_pool{s}'),
            tags={f'K_pool{s}'}
        )

        return qtn.TensorNetwork([C_w_tn, C_h_tn, parameters_t])
    
    # This method converts the flatten layer to an equivalent tensor network
    def convert_flatten(self):
        """
        Builds a tensor network implementing:
            (1, RMS-TERM, data) -> [RMS-value, flatten(data)]
        """
        channels, height, width = self.current_size
        assert channels >= 2, "convert_flatten expects layout (1, RMS-TERM, data...)"
        data_channels = channels - 2
        output_length = 1 + data_channels * height * width

        F = torch.zeros(output_length, channels, height, width)

        # RMS-TERM is a uniform matrix consisting of the same value RMS-TERM, we extract one (at h = w = 0) and put it in the output
        F[0, 1, 0, 0] = 1.0

        # We flatten the data channels
        for c in range(data_channels):
            for h in range(height):
                for w in range(width):
                    # Formula to know the index i of some input (c, h, w) in a vector
                    i = 1 + c * (height * width) + h * width + w
                    F[i, 2 + c, h, w] = 1.0

        F = qtn.Tensor(
            data=F,
            inds=('flatten', 'C_in', 'H_in', 'W_in')
        )

        c, h, w = self.current_size
        self.current_size = (1 + ((c - 2) * h * w),)
        return F
    
    # This methods converts the linear layer to an equivalent tensor network
    def convert_linear(self, linear: Linear):
        """
        Builds a tensor network implementing:
            [RMS-value, flattened_data] -> [RMS-value, linear(flattened_data)]
        """
        assert linear.bias is None, "Bias not supported in conversion"
        in_data_features = self.current_size[0] - 1
        assert in_data_features == linear.in_features, (
            f"Mismatch: expected in_features={linear.in_features} "
            f"(current_size {self.current_size[0] - 1})"
        )

        in_features = linear.in_features
        out_features = linear.out_features

        # The resulting matrix consists of the weight matrix, extended with a 1 to make sure the RMS-value pass through
        L = torch.zeros(1 + out_features, 1 + in_features)
        L[0, 0] = 1.0
        L[1:, 1:] = linear.weight.detach()

        self.current_size = (1 + out_features,)

        return qtn.Tensor(
            data=L,
            inds=('linear_out', 'flatten')
        )
    
    def convert_adaptive_avg_pool_2d(self, adaptive: AdaptiveAvgPool2d, unique_label: str = ''):
        """
        Builds a tensor network implementing:
            1, RMS-TERM, data) -> (1, RMS-TERM, AdaptiveAvgPool(data))
        """
        assert adaptive.output_size == 1, f"Output size {adaptive.output_size} not supported; only 1 is supported"

        total_channels, H, W = self.current_size
        self.current_size = (total_channels, 1, 1)

        C_h = torch.eye(H).unsqueeze(0)  # (1, H, H)
        C_w = torch.eye(W).unsqueeze(0)  # (1, W, W)

        # Uniform 1/(H*W) across all kernel positions
        with torch.no_grad():
            parameters = torch.zeros(total_channels, total_channels, H, W)
            diag = torch.arange(total_channels)
            parameters[diag, diag, :, :] = 1.0 / float(H * W)

        s = f'_{unique_label}' if unique_label else ''
        in_label = f'_in{s}' if unique_label else '_combined'
        out_label = f'_out{s}' if unique_label else '_out'

        C_h_tn = qtn.Tensor(
            data=C_h,
            inds=(f'H{out_label}', f'H{in_label}', f'K_h_gap{s}'),
            tags={f'C_h_gap{s}'}
        )
        C_w_tn = qtn.Tensor(
            data=C_w,
            inds=(f'W{out_label}', f'W{in_label}', f'K_w_gap{s}'),
            tags={f'C_w_gap{s}'}
        )
        params_t = qtn.Tensor(
            data=parameters,
            inds=(f'C{out_label}', f'C{in_label}', f'K_h_gap{s}', f'K_w_gap{s}'),
            tags={f'K_gap{s}'}
        )

        return qtn.TensorNetwork([C_h_tn, C_w_tn, params_t])
