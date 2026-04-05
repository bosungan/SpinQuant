# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import logging
import math
import pprint
import time

import torch
import torch.nn as nn
import tqdm

from utils import quant_utils, utils

import torch

import torch

def pack_int4_to_int32(w_int: torch.Tensor) -> torch.Tensor():
    """
    Pack signed int4 values [-8, 7] into int32 along the last dimension.
    8 values per int32 (AWQ format).

    Input:
        w_int: integer tensor with shape (out_features, in_features), values in [-8, 7]

    Output:
        packed: int32 tensor with shape (out_features, in_features // 8)
    """
    assert w_int.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
    assert torch.all((w_int >= -8) & (w_int <= 7)), "Values must be in [-8, 7]"

    last_dim = w_int.shape[-1]
    assert last_dim % 8 == 0, f"Last dimension must be divisible by 8, got {last_dim}"

    # signed int4 [-8,7] → unsigned nibble [0,15]
    x_u4 = (w_int.to(torch.int32) & 0xF)  # (out, in)

    # (out, in//8, 8) 로 reshape 후 각 위치에 shift
    x_grouped = x_u4.reshape(*w_int.shape[:-1], last_dim // 8, 8)  # (..., in//8, 8)

    # bit 0~3: val[0], bit 4~7: val[1], ..., bit 28~31: val[7]
    shifts = torch.arange(0, 32, 4, device=w_int.device, dtype=torch.int32)  # [0,4,8,...,28]
    packed = (x_grouped << shifts).sum(dim=-1).to(torch.int32)  # (..., in//8)

    return packed


def compress_groupwise_scale(scale: torch.Tensor, groupsize: int, atol=0.0, rtol=0.0):
    """
    scale: repeated scale tensor, e.g. shape (out_features, in_features)
           where values are repeated along the last dimension per groupsize.
    groupsize: number of consecutive elements sharing one scale
    atol, rtol: tolerance for equality check

    Returns:
        compressed_scale: shape (..., last_dim // groupsize)
    Raises:
        ValueError if the repeated pattern is not satisfied.
    """
    if scale.ndim < 1:
        raise ValueError(f"scale must have at least 1 dimension, got shape={scale.shape}")

    last_dim = scale.shape[-1]
    if last_dim % groupsize != 0:
        raise ValueError(
            f"Last dim {last_dim} is not divisible by groupsize {groupsize}"
        )

    num_groups = last_dim // groupsize

    # (..., num_groups, groupsize)
    reshaped = scale.reshape(*scale.shape[:-1], num_groups, groupsize)

    # representative value from each group
    compressed = reshaped[..., 0]   # (..., num_groups)

    # check every value in each group equals representative
    ref = compressed.unsqueeze(-1)  # (..., num_groups, 1)
    if not torch.allclose(reshaped, ref, atol=atol, rtol=rtol):
        diff = (reshaped - ref).abs().max().item()
        raise ValueError(
            f"Scale is not constant within each group of size {groupsize}. "
            f"Max abs diff inside group = {diff}"
        )

    return compressed

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Scale1 = torch.zeros_like(W1).to(Scale.dtype)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q, int_weight, scale, zero = self.quantizer.fake_quantize(w.unsqueeze(1))
                Q1[:, i] = q.flatten()
                q = q.flatten()
                W_int1[:, i] = int_weight.flatten()
                Scale1[:, i] = scale.flatten()

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Scale[:, i1:i2] = Scale1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        # if export_to_et:
        #     self.layer.register_buffer(
        #         "int_weight", W_int.reshape(self.layer.weight.shape)
        #     )
        #     self.layer.register_buffer("scale", Scale)
        
        # always register int_weight and scale
        W_int = W_int.to(torch.int8)        
        W_int_T = W_int.T.contiguous()  # (in_features, out_features)
        int_weight_packed = pack_int4_to_int32(W_int_T)  # (in_features, out_features//8)
        assert int_weight_packed.shape == (W_int.shape[1], W_int.shape[0] // 8)
        self.layer.register_buffer("qweight", int_weight_packed)

        compressed_scale = compress_groupwise_scale(Scale, groupsize).half()
        self.layer.register_buffer("scales", compressed_scale.T.contiguous())  # (n_groups, out)
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]
    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args, custom_layers=None):
    """
    From GPTQ repo
    """
    # assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    if custom_layers:
        layers = custom_layers
    else:
        layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm (range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, torch.nn.Embedding]
        )

        for name in subset:
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
            if args.export_to_et:
                layer_weight_bits = 8  # all per channel 8 bits for executorch export
                w_groupsize = -1
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale, zero = quantizer.fake_quantize(W)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)        
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers