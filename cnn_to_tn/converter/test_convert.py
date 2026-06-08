import unittest

import torch
import torch.nn as nn
import quimb.tensor as qtn

from converter.convert import Convert
from utils.BilinearLayer2d import BilinearLayer2d
from utils.RMSNorm2d import RMSNorm2d
from utils.ResidualBlock import ResidualBlock
from utils.model import BaseCNN

# Methods to build the input
def cat_ch(*tensors):
    return torch.cat(list(tensors), dim=0)

def ones_ch(H, W):
    return torch.ones(1, H, W)

def scalar_ch(value, H, W):
    return torch.full((1, H, W), float(value))

def bilinear_inputs(tensor, suf=""):
    s = f"_{suf}" if suf else ""
    left  = qtn.Tensor(tensor, inds=(f"C_in_L{s}", f"H_in_L{s}", f"W_in_L{s}"))
    right = qtn.Tensor(tensor, inds=(f"C_in_R{s}", f"H_in_R{s}", f"W_in_R{s}"))
    return qtn.TensorNetwork([left, right])

def contract_bilinear(network, inp, suf=""):
    s = f"_{suf}" if suf else ""
    out_inds = (f"C_out{s}", f"H_out{s}", f"W_out{s}")
    return (network & bilinear_inputs(inp, suf)).contract(output_inds=out_inds).data

def linear_input(tensor, c="C_combined", h="H_combined", w="W_combined"):
    return qtn.Tensor(tensor, inds=(c, h, w))

def contract_linear(network, inp,
                    c_in="C_combined", h_in="H_combined", w_in="W_combined",
                    c_out="C_out", h_out="H_out", w_out="W_out"):
    return (network & linear_input(inp, c_in, h_in, w_in)).contract(
        output_inds=(c_out, h_out, w_out)
    ).data

def inputs_for_open_inds(network, tensor):
    outer = set(network.outer_inds())
    c_inds = sorted([i for i in outer if i.startswith("C_in")])

    inputs = []
    used = set()
    for c_ind in c_inds:
        suffix = c_ind[len("C_in") :]
        h_ind = f"H_in{suffix}"
        w_ind = f"W_in{suffix}"
        if h_ind not in outer or w_ind not in outer:
            raise AssertionError(f"Missing H/W indices for input suffix '{suffix}'")
        inputs.append(qtn.Tensor(tensor, inds=(c_ind, h_ind, w_ind)))
        used.update({c_ind, h_ind, w_ind})

    unmatched = {i for i in outer if i.startswith(("C_in", "H_in", "W_in"))} - used
    if unmatched:
        raise AssertionError(f"Unmatched input indices: {unmatched}")

    return qtn.TensorNetwork(inputs)

# Stepwise contraction of a tensor network, it avoids to construct exponentially wide tensor networks when not needed
def contract_model_stepwise(model, inp):
    converter = Convert(tuple(inp.shape))
    current = inp

    number_of_residual_blocks = model.num_stages * model.blocks_per_stage
    residual_block_count = 1

    for layer in model.layers:
        if isinstance(layer, ResidualBlock):
            double_copy = not (residual_block_count == number_of_residual_blocks)
            block_tn = converter.convert_residual_block(layer, double_copy=double_copy)
            current = (block_tn & inputs_for_open_inds(block_tn, current)).contract(
                output_inds=("C_out", "H_out", "W_out")
            ).data
            rms = current[1, 0, 0]
            current[2:] = current[2:] / rms
            current[1, :, :] = 1
            residual_block_count += 1
        elif isinstance(layer, nn.AvgPool2d):
            pool_tn = converter.convert_avgpool2d(layer)
            current = contract_linear(pool_tn, current)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            gap_tn = converter.convert_adaptive_avg_pool_2d(layer)
            current = contract_linear(gap_tn, current)
        elif isinstance(layer, nn.Flatten):
            flatten_t = converter.convert_flatten()
            current = (flatten_t & linear_input(current, "C_in", "H_in", "W_in")).contract(
                output_inds=("flatten",)
            ).data
        elif isinstance(layer, nn.Linear):
            linear_t = converter.convert_linear(layer)
            current = (linear_t & qtn.Tensor(current, inds=("flatten",))).contract(
                output_inds=("linear_out",)
            ).data
        else:
            raise AssertionError(
                f"Unsupported layer in model stepwise conversion: {type(layer)}"
            )
    return current

# ─── tests ───────────────────────────────────────────────────────────────────

H, W = 4, 5


class TestConvertRms(unittest.TestCase):

    def test_rms_term_and_gamma_scaling(self):
        B, C = 3, 2
        rms_layer = RMSNorm2d(B, mode="global")
        rms_layer.gamma = nn.Parameter(torch.rand(1, B, 1, 1))
        gamma = rms_layer.gamma.detach().reshape(-1)

        rms_prev  = torch.rand(1, H, W)
        bilinear  = torch.rand(B, H, W)
        copy_data = torch.rand(C, H, W)
        inp = cat_ch(ones_ch(H, W), rms_prev, bilinear, copy_data)

        converter = Convert((1 + 1 + B + C, H, W))
        network = converter.convert_rms(rms_layer, B, C)
        out = contract_bilinear(network, inp)

        rms_new = (bilinear ** 2).mean()
        expected = cat_ch(
            ones_ch(H, W),
            scalar_ch(rms_new, H, W),
            rms_prev,
            gamma.view(-1, 1, 1) * bilinear,
            copy_data,
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_constant_channel_preserved(self):
        B, C = 2, 2
        rms_layer = RMSNorm2d(B, mode="global")
        bilinear  = torch.rand(B, H, W)
        copy_data = torch.rand(C, H, W)
        inp = cat_ch(ones_ch(H, W), torch.rand(1, H, W), bilinear, copy_data)

        converter = Convert((1 + 1 + B + C, H, W))
        network = converter.convert_rms(rms_layer, B, C)
        out = contract_bilinear(network, inp)

        self.assertTrue(torch.allclose(out[0], torch.ones(H, W), atol=1e-5))

    def test_rms_term_is_global_mean_square(self):
        B, C = 4, 1
        rms_layer = RMSNorm2d(B, mode="global")
        bilinear  = torch.rand(B, H, W)
        copy_data = torch.rand(C, H, W)
        inp = cat_ch(ones_ch(H, W), torch.rand(1, H, W), bilinear, copy_data)

        converter = Convert((1 + 1 + B + C, H, W))
        network = converter.convert_rms(rms_layer, B, C)
        out = contract_bilinear(network, inp)

        expected_rms = (bilinear ** 2).mean()
        self.assertTrue(torch.allclose(out[1], torch.full((H, W), expected_rms.item()), atol=1e-5))


class TestConvertGamma(unittest.TestCase):

    def test_gamma_scales_bilinear_channels(self):
        B, C = 3, 2
        rms_layer = RMSNorm2d(B, mode="global")
        rms_layer.gamma = nn.Parameter(torch.rand(1, B, 1, 1))
        gamma = rms_layer.gamma.detach().reshape(-1)

        rms_term  = torch.rand(1, H, W)
        bilinear  = torch.rand(B, H, W)
        copy_data = torch.rand(C, H, W)
        inp = cat_ch(ones_ch(H, W), rms_term, bilinear, copy_data)

        converter = Convert((1 + 1 + B + C, H, W))
        network = converter.convert_gamma(rms_layer, B, C)
        out = contract_bilinear(network, inp)

        expected = cat_ch(
            ones_ch(H, W),
            rms_term,
            gamma.view(-1, 1, 1) * bilinear,
            copy_data,
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_non_bilinear_channels_unchanged(self):
        B, C = 3, 2
        rms_layer = RMSNorm2d(B, mode="global")
        rms_layer.gamma = nn.Parameter(torch.rand(1, B, 1, 1))

        rms_term  = torch.rand(1, H, W)
        bilinear  = torch.rand(B, H, W)
        copy_data = torch.rand(C, H, W)
        inp = cat_ch(ones_ch(H, W), rms_term, bilinear, copy_data)

        converter = Convert((1 + 1 + B + C, H, W))
        network = converter.convert_gamma(rms_layer, B, C)
        out = contract_bilinear(network, inp)

        self.assertTrue(torch.allclose(out[0],  torch.ones(H, W), atol=1e-5), "constant channel changed")
        self.assertTrue(torch.allclose(out[1],  rms_term.squeeze(0), atol=1e-5), "RMS-TERM changed")
        self.assertTrue(torch.allclose(out[2+B:], copy_data, atol=1e-5), "copy_input changed")


class TestConvertResidualConnection(unittest.TestCase):

    def _run(self, n, double_copy=False):
        rms1_val, rms2_val = 0.5, 0.7
        rms1      = scalar_ch(rms1_val, H, W)
        rms2      = scalar_ch(rms2_val, H, W)
        bilinear  = torch.rand(n, H, W)
        copy_data = torch.rand(n, H, W)
        inp = cat_ch(ones_ch(H, W), rms1, rms2, bilinear, copy_data)

        converter = Convert((3 + 2 * n, H, W))
        network = converter.convert_residual_connection(n, double_copy=double_copy)
        return contract_bilinear(network, inp), bilinear, copy_data, rms1_val, rms2_val

    def test_residual_sum(self):
        out, bilinear, copy_data, rms1_val, rms2_val = self._run(n=3)
        expected_result = rms2_val * bilinear + rms1_val * copy_data
        self.assertTrue(torch.allclose(out[2:], expected_result, atol=1e-5))

    def test_rms_product(self):
        out, _, _, rms1_val, rms2_val = self._run(n=3)
        self.assertTrue(torch.allclose(out[1], torch.full((H, W), rms1_val * rms2_val), atol=1e-5))

    def test_double_copy_produces_two_identical_result_blocks(self):
        n = 3
        out, bilinear, copy_data, rms1_val, rms2_val = self._run(n=n, double_copy=True)
        result = rms2_val * bilinear + rms1_val * copy_data
        self.assertTrue(torch.allclose(out[2:2+n], result, atol=1e-5))
        self.assertTrue(torch.allclose(out[2+n:],  result, atol=1e-5))

    def test_output_size_no_double_copy(self):
        n = 3
        out, *_ = self._run(n=n, double_copy=False)
        self.assertEqual(out.shape[0], 2 + n)

    def test_output_size_double_copy(self):
        n = 3
        out, *_ = self._run(n=n, double_copy=True)
        self.assertEqual(out.shape[0], 2 + 2 * n)

    def test_constant_channel_preserved(self):
        out, *_ = self._run(n=3)
        self.assertTrue(torch.allclose(out[0], torch.ones(H, W), atol=1e-5))

    def test_n_equals_one(self):
        rms1_val, rms2_val = 0.3, 0.8
        bilinear  = torch.rand(1, H, W)
        copy_data = torch.rand(1, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(rms1_val, H, W), scalar_ch(rms2_val, H, W),
                     bilinear, copy_data)
        converter = Convert((5, H, W))
        network = converter.convert_residual_connection(1)
        out = contract_bilinear(network, inp)

        self.assertTrue(torch.allclose(out[2], rms2_val * bilinear[0] + rms1_val * copy_data[0], atol=1e-5))
        self.assertTrue(torch.allclose(out[1], torch.full((H, W), rms1_val * rms2_val), atol=1e-5))

    def test_unique_label_changes_indices(self):
        n = 2
        rms1_val, rms2_val = 0.4, 0.6
        bilinear  = torch.rand(n, H, W)
        copy_data = torch.rand(n, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(rms1_val, H, W), scalar_ch(rms2_val, H, W),
                     bilinear, copy_data)

        converter = Convert((3 + 2 * n, H, W))
        network = converter.convert_residual_connection(n, unique_label="r1")
        out = contract_bilinear(network, inp, suf="r1")

        expected = rms2_val * bilinear + rms1_val * copy_data
        self.assertTrue(torch.allclose(out[2:], expected, atol=1e-5))


class TestConvertResidualBlock(unittest.TestCase):

    def _contract_block(self, network, inp):
        input_tn = inputs_for_open_inds(network, inp)
        return (network & input_tn).contract(
            output_inds=("C_out", "H_out", "W_out")
        ).data

    def test_identity_skip(self):
        in_channels = 3
        block = ResidualBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            upscale_factor=2,
            kernel_size=3,
            repeat_factor=1,
            rms_norm_mode="global",
        )
        block.eval()

        rms_layer = block.layers[0]
        rms_layer.gamma = nn.Parameter(torch.rand(1, in_channels, 1, 1))

        rms_prev_val = 0.6
        data = torch.rand(in_channels, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(rms_prev_val, H, W), data, data)

        converter = Convert((2 + 2 * in_channels, H, W))
        network = converter.convert_residual_block(block)
        out = self._contract_block(network, inp)

        gamma = rms_layer.gamma.detach().reshape(-1)
        rms_new = (data ** 2).mean()
        bilinear_out = block.layers[1]((gamma.view(-1, 1, 1) * data).unsqueeze(0)).squeeze(0)
        skip_out = data
        result = rms_prev_val * bilinear_out + rms_new * skip_out

        expected = cat_ch(
            ones_ch(H, W),
            scalar_ch(rms_new, H, W) * rms_prev_val,
            result,
            result,
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_conv_skip(self):
        in_channels = 3
        out_channels = 4
        block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=2,
            kernel_size=3,
            repeat_factor=1,
            rms_norm_mode="global",
        )
        block.eval()

        rms_layer = block.layers[0]
        rms_layer.gamma = nn.Parameter(torch.rand(1, in_channels, 1, 1))

        rms_prev_val = 0.4
        data = torch.rand(in_channels, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(rms_prev_val, H, W), data, data)

        converter = Convert((2 + 2 * in_channels, H, W))
        network = converter.convert_residual_block(block)
        out = self._contract_block(network, inp)

        gamma = rms_layer.gamma.detach().reshape(-1)
        rms_new = (data ** 2).mean()
        bilinear_out = block.layers[1]((gamma.view(-1, 1, 1) * data).unsqueeze(0)).squeeze(0)
        skip_out = block.skip(data.unsqueeze(0)).squeeze(0)
        result = rms_prev_val * bilinear_out + rms_new * skip_out

        expected = cat_ch(
            ones_ch(H, W),
            scalar_ch(rms_new, H, W) * rms_prev_val,
            result,
            result,
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_residual_output_reconstructs_from_tn_rms(self):
        in_channels = 3
        out_channels = 4
        block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=2,
            kernel_size=3,
            repeat_factor=4,
            rms_norm_mode="global",
        )
        block.eval()

        rms_layer = block.layers[0]
        rms_layer.gamma = nn.Parameter(torch.rand(1, in_channels, 1, 1))
        rms_layer.epsilon = 0.0

        data = torch.rand(in_channels, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(1.0, H, W), data, data)

        converter = Convert((2 + 2 * in_channels, H, W))
        tn = converter.convert_residual_block(block)
        tn_out = (tn & inputs_for_open_inds(tn, inp)).contract(
            output_inds=("C_out", "H_out", "W_out")
        ).data
        
        rms_term = tn_out[1]
        expected = (1 / rms_term) * tn_out[2:2 + out_channels]

        actual = block(data.unsqueeze(0)).squeeze(0)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class TestConvertBilinearLayer(unittest.TestCase):

    def _run(self, rms_count=1):
        B_in, B_out, upscale, C = 3, 4, 2, 2
        bl = BilinearLayer2d(in_channels=B_in, upscale_factor=upscale, out_channels=B_out, kernel_size=3)

        rms_terms = [torch.rand(1, H, W) for _ in range(rms_count)]
        bilinear  = torch.rand(B_in, H, W)
        copy_data = torch.rand(C, H, W)
        inp = cat_ch(ones_ch(H, W), *rms_terms, bilinear, copy_data)

        converter = Convert((1 + rms_count + B_in + C, H, W))
        network = converter.convert_bilinear_layer(bl, copy_input_size=C, rms_count=rms_count)
        out = contract_bilinear(network, inp)

        with torch.no_grad():
            bl_out = bl(bilinear.unsqueeze(0)).squeeze(0)

        return out, bl_out, rms_terms, copy_data, B_out, C

    def test_bilinear_output_matches_pytorch(self):
        out, bl_out, rms_terms, _, B_out, _ = self._run(rms_count=1)
        rms_end = 1 + 1  # const + 1 rms term
        self.assertTrue(torch.allclose(out[rms_end:rms_end + B_out], bl_out, atol=1e-5))

    def test_rms_terms_pass_through(self):
        out, _, rms_terms, _, _, _ = self._run(rms_count=1)
        self.assertTrue(torch.allclose(out[1], rms_terms[0].squeeze(0), atol=1e-5))

    def test_copy_input_passes_through(self):
        out, _, _, copy_data, B_out, C = self._run(rms_count=1)
        self.assertTrue(torch.allclose(out[-(C):], copy_data, atol=1e-5))

    def test_two_rms_terms(self):
        out, bl_out, rms_terms, copy_data, B_out, C = self._run(rms_count=2)
        self.assertTrue(torch.allclose(out[1], rms_terms[0].squeeze(0), atol=1e-5))
        self.assertTrue(torch.allclose(out[2], rms_terms[1].squeeze(0), atol=1e-5))
        self.assertTrue(torch.allclose(out[3:3 + B_out], bl_out, atol=1e-5))
        self.assertTrue(torch.allclose(out[-C:], copy_data, atol=1e-5))


class TestConvertAvgPool2d(unittest.TestCase):

    def _run(self, pool, H_in=8, W_in=8, C_data=3, unique_label=''):
        total_C = 1 + 1 + C_data
        inp = torch.rand(total_C, H_in, W_in)
        inp[0] = 1.0                          # constant-1
        inp[1] = 0.5                          # spatially uniform RMS-TERM

        converter = Convert((total_C, H_in, W_in))
        network = converter.convert_avgpool2d(pool, unique_label=unique_label)

        if unique_label:
            s = f'_{unique_label}'
            out = contract_linear(
                network, inp,
                c_in=f'C_in{s}', h_in=f'H_in{s}', w_in=f'W_in{s}',
                c_out=f'C_out{s}', h_out=f'H_out{s}', w_out=f'W_out{s}',
            )
        else:
            out = contract_linear(network, inp)

        with torch.no_grad():
            expected = pool(inp.unsqueeze(0)).squeeze(0)
        return out, expected, inp

    def test_matches_pytorch_basic(self):
        out, expected, _ = self._run(nn.AvgPool2d(kernel_size=2, stride=2))
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_matches_pytorch_with_padding(self):
        out, expected, _ = self._run(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_matches_pytorch_asymmetric_kernel(self):
        out, expected, _ = self._run(nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 2)), H_in=8, W_in=8)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_constant_channel_preserved(self):
        out, _, inp = self._run(nn.AvgPool2d(kernel_size=2, stride=2))
        self.assertTrue(torch.allclose(out[0], torch.ones_like(out[0]), atol=1e-5))

    def test_uniform_rms_term_preserved(self):
        out, _, inp = self._run(nn.AvgPool2d(kernel_size=2, stride=2))
        self.assertTrue(torch.allclose(out[1], torch.full_like(out[1], 0.5), atol=1e-5))

    def test_channels_not_mixed(self):
        # Each output channel must depend only on the same input channel (depthwise)
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        total_C = 1 + 1 + 2
        inp = torch.zeros(total_C, 4, 4)
        inp[0] = 1.0
        inp[2] = 9.0   # data channel 0 set to a distinct constant
        inp[3] = 0.0   # data channel 1 stays 0

        converter = Convert((total_C, 4, 4))
        network = converter.convert_avgpool2d(pool)
        out = contract_linear(network, inp)

        self.assertTrue(torch.allclose(out[2], torch.full_like(out[2], 9.0), atol=1e-5))
        self.assertTrue(torch.allclose(out[3], torch.zeros_like(out[3]), atol=1e-5))

    def test_unique_label_changes_indices(self):
        out, expected, _ = self._run(nn.AvgPool2d(kernel_size=2, stride=2), unique_label='p')
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


class TestConvertAdaptiveAvgPool2d(unittest.TestCase):

    def test_matches_pytorch(self):
        C_data = 3
        total_C = 1 + 1 + C_data
        adaptive = nn.AdaptiveAvgPool2d(1)
        inp = torch.rand(total_C, H, W)
        inp[0] = 1.0

        converter = Convert((total_C, H, W))
        network = converter.convert_adaptive_avg_pool_2d(adaptive)
        out = contract_linear(network, inp)

        with torch.no_grad():
            expected = adaptive(inp.unsqueeze(0)).squeeze(0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


class TestConvertFlatten(unittest.TestCase):

    def _run(self, C_data, rms_val, data, H_in=H, W_in=W):
        total_C = 1 + 1 + C_data
        inp = cat_ch(ones_ch(H_in, W_in), scalar_ch(rms_val, H_in, W_in), data)
        converter = Convert((total_C, H_in, W_in))
        flatten_t = converter.convert_flatten()
        return (flatten_t & linear_input(inp, "C_in", "H_in", "W_in")).contract(
            output_inds=("flatten",)
        ).data

    def test_rms_term_placed_first(self):
        rms_val = 0.42
        data = torch.rand(3, H, W)
        out = self._run(3, rms_val, data)
        self.assertAlmostEqual(out[0].item(), rms_val, places=5)

    def test_constant_channel_dropped(self):
        # Put a large distinct value in the constant channel; it must not appear in output
        rms_val = 0.1
        data = torch.rand(2, H, W)
        total_C = 1 + 1 + 2
        inp = cat_ch(torch.full((1, H, W), 999.0), scalar_ch(rms_val, H, W), data)
        converter = Convert((total_C, H, W))
        flatten_t = converter.convert_flatten()
        out = (flatten_t & linear_input(inp, "C_in", "H_in", "W_in")).contract(
            output_inds=("flatten",)
        ).data
        self.assertFalse(torch.any(out > 100).item(), "Constant channel leaked into output")

    def test_c_major_data_ordering(self):
        C_data = 3
        # Assign unique values so any permutation would produce a different result
        data = torch.arange(C_data * H * W, dtype=torch.float32).reshape(C_data, H, W)
        out = self._run(C_data, 0.0, data)
        expected = torch.cat([torch.tensor([0.0]), data.flatten()])
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_output_length(self):
        C_data = 3
        out = self._run(C_data, 0.1, torch.rand(C_data, H, W))
        self.assertEqual(out.shape[0], 1 + C_data * H * W)

    def test_rms_picks_h0_w0_value(self):
        # RMS-TERM channel is spatially uniform, but the implementation picks (h=0, w=0)
        rms_val = 0.77
        data = torch.rand(2, H, W)
        out = self._run(2, rms_val, data)
        self.assertAlmostEqual(out[0].item(), rms_val, places=5)


class TestConvertLinear(unittest.TestCase):

    def test_matches_pytorch(self):
        C_data = 4
        in_features = C_data * H * W
        out_features = 10
        rms_val = 0.3

        data = torch.rand(in_features)
        linear = nn.Linear(in_features, out_features, bias=False)

        flat_data = torch.cat([torch.tensor([rms_val]), data.flatten()])
        flat_t = qtn.Tensor(flat_data, inds=("flatten",))

        converter = Convert((in_features + 1,))
        linear_t = converter.convert_linear(linear)
        out = (linear_t & flat_t).contract(output_inds=("linear_out",)).data

        with torch.no_grad():
            expected = torch.cat([
                torch.tensor([rms_val]),
                (linear.weight @ data.flatten()).detach(),
            ])
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_rms_term_passes_through(self):
        C_data = 2
        in_features = C_data * H * W
        rms_val = 0.99
        data = torch.rand(in_features)
        linear = nn.Linear(C_data * H * W, 5, bias=False)

        flat_data = torch.cat([torch.tensor([rms_val]), data.flatten()])
        flat_t = qtn.Tensor(flat_data, inds=("flatten",))

        converter = Convert((in_features + 1,))
        linear_t = converter.convert_linear(linear)
        out = (linear_t & flat_t).contract(output_inds=("linear_out",)).data

        self.assertAlmostEqual(out[0].item(), rms_val, places=5)


class TestConvertConv2d(unittest.TestCase):

    def _build_input(self, C_meta, C_in):
        meta = torch.rand(C_meta, H, W)
        data = torch.rand(C_in, H, W)
        return torch.cat([meta, data], dim=0), meta, data

    def test_data_channels_convolved(self):
        C_meta, C_in, C_out = 2, 3, 4
        conv = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False)
        inp, meta, data = self._build_input(C_meta, C_in)

        converter = Convert((C_meta + C_in, H, W))
        network = converter.convert_conv2d_extended(conv, data_start=C_meta, data_len=C_in, unique_label="", to_hidden=False)
        out = contract_linear(network, inp)

        with torch.no_grad():
            conv_out = conv(data.unsqueeze(0)).squeeze(0)
        self.assertTrue(torch.allclose(out[C_meta:], conv_out, atol=1e-5))

    def test_meta_channels_pass_through(self):
        C_meta, C_in = 2, 3
        conv = nn.Conv2d(C_in, C_in, kernel_size=3, padding=1, bias=False)
        inp, meta, _ = self._build_input(C_meta, C_in)

        converter = Convert((C_meta + C_in, H, W))
        network = converter.convert_conv2d_extended(conv, data_start=C_meta, data_len=C_in, unique_label="", to_hidden=False)
        out = contract_linear(network, inp)

        self.assertTrue(torch.allclose(out[:C_meta], meta, atol=1e-5))

    def test_to_hidden_true_data_transformed(self):
        C_meta, C_in, C_hidden = 2, 3, 6
        conv = nn.Conv2d(C_in, C_hidden, kernel_size=3, padding=1, bias=False)
        inp, meta, data = self._build_input(C_meta, C_in)

        converter = Convert((C_meta + C_in, H, W))
        network = converter.convert_conv2d_extended(conv, data_start=C_meta, data_len=C_in, unique_label="L", to_hidden=True)
        # to_hidden=True uses C_in_L/H_in_L/W_in_L -> C_hidden_L/H_hidden_L/W_hidden_L
        out = contract_linear(
            network, inp,
            c_in="C_in_L", h_in="H_in_L", w_in="W_in_L",
            c_out="C_hidden_L", h_out="H_hidden_L", w_out="W_hidden_L",
        )

        with torch.no_grad():
            conv_out = conv(data.unsqueeze(0)).squeeze(0)
        self.assertTrue(torch.allclose(out[C_meta:], conv_out, atol=1e-5))

    def test_to_hidden_true_meta_passes_through(self):
        C_meta, C_in, C_hidden = 2, 3, 6
        conv = nn.Conv2d(C_in, C_hidden, kernel_size=3, padding=1, bias=False)
        inp, meta, _ = self._build_input(C_meta, C_in)

        converter = Convert((C_meta + C_in, H, W))
        network = converter.convert_conv2d_extended(conv, data_start=C_meta, data_len=C_in, unique_label="L", to_hidden=True)
        out = contract_linear(
            network, inp,
            c_in="C_in_L", h_in="H_in_L", w_in="W_in_L",
            c_out="C_hidden_L", h_out="H_hidden_L", w_out="W_hidden_L",
        )
        self.assertTrue(torch.allclose(out[:C_meta], meta, atol=1e-5))

    def test_different_meta_channel_counts(self):
        for C_meta in [1, 3, 4]:
            with self.subTest(C_meta=C_meta):
                C_in = 3
                conv = nn.Conv2d(C_in, C_in, kernel_size=3, padding=1, bias=False)
                inp, meta, data = self._build_input(C_meta, C_in)

                converter = Convert((C_meta + C_in, H, W))
                network = converter.convert_conv2d_extended(conv, data_start=C_meta, data_len=C_in, unique_label="", to_hidden=False)
                out = contract_linear(network, inp)

                self.assertTrue(torch.allclose(out[:C_meta], meta, atol=1e-5))


class TestConvertModel(unittest.TestCase):

    def test_convert_model_three_stages_three_blocks(self):
        in_channels = 3
        num_classes = 10

        class Cfg:
            pass

        cfg = Cfg()
        cfg.num_stages = 4
        cfg.blocks_per_stage = 3
        cfg.upscale_factor = 2
        cfg.channels_factor = 2
        cfg.start_output_channels = in_channels
        cfg.kernel_size = 3
        cfg.repeat_factor = 4
        cfg.rms_norm_mode = "global"

        model = BaseCNN(cfg, in_channels=in_channels, input_size=H, num_classes=num_classes)
        model.eval()

        # Ensure RMS gamma params exist for all RMSNorm2d layers
        for layer in model.layers:
            if isinstance(layer, ResidualBlock):
                for sub in layer.layers:
                    if isinstance(sub, RMSNorm2d):
                        gamma = sub.gamma
                        sub.gamma = nn.Parameter(torch.rand_like(gamma.data))

        data = torch.rand(in_channels, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(1.0, H, W), data, data)

        tn_out = contract_model_stepwise(model, inp)

        with torch.no_grad():
            expected = model(data.unsqueeze(0)).squeeze(0)

        output = tn_out[1:] / tn_out[0]

        self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    def test_convert_model_two_stages_two_blocks(self):
        in_channels = 3
        num_classes = 5

        class Cfg:
            pass

        cfg = Cfg()
        cfg.num_stages = 2
        cfg.blocks_per_stage = 2
        cfg.upscale_factor = 2
        cfg.channels_factor = 2
        cfg.start_output_channels = in_channels
        cfg.kernel_size = 3
        cfg.repeat_factor = 1
        cfg.rms_norm_mode = "global"

        model = BaseCNN(cfg, in_channels=in_channels, input_size=H, num_classes=num_classes)
        model.eval()

        # Ensure RMS gamma params exist for all RMSNorm2d layers
        for layer in model.layers:
            if isinstance(layer, ResidualBlock):
                for sub in layer.layers:
                    if isinstance(sub, RMSNorm2d):
                        gamma = sub.gamma
                        sub.gamma = nn.Parameter(torch.rand_like(gamma.data))

        data = torch.rand(in_channels, H, W)
        inp = cat_ch(ones_ch(H, W), scalar_ch(1.0, H, W), data, data)

        converter = Convert((2 + 2 * in_channels, H, W))
        tn = converter.convert_model(model)

        input_tn = inputs_for_open_inds(tn, inp)
        tn_out = (tn & input_tn).contract(output_inds=("linear_out",)).data

        with torch.no_grad():
            expected = model(data.unsqueeze(0)).squeeze(0)

        output = tn_out[1:] / tn_out[0]

        self.assertTrue(torch.allclose(output, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
