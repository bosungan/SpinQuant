# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)

from logging import Logger
log: Logger = utils.get_logger("spinquant")

def remove_fp_module_weights(state_dict):
    keep_patterns = [
        "embed_tokens.weight",
        "lm_head.weight", 
        "norm.weight",
        "layernorm",
    ]
    keys_to_remove = [
        k for k in state_dict.keys()
        if (k.endswith(".module.weight") or k.endswith(".weight"))
        and not any(pattern in k for pattern in keep_patterns)
    ]
    for k in keys_to_remove:
        del state_dict[k]
    return state_dict


def remove_quantizer_weights(state_dict):
    keys_to_remove = [
        k for k in state_dict.keys()
        if "quantizer." in k or "had_K" in k  # had_K 추가
    ]
    for k in keys_to_remove:
        del state_dict[k]
    return state_dict

def convert_keys_to_awq_format(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        # .module. 제거
        new_k = k 
        new_k = new_k.replace(".module.", ".")
        new_dict[new_k] = v
    return new_dict

def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    
    # Rotate the weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            # save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        if args.save_qmodel_path:
            save_dict = model.state_dict()
            save_dict = remove_fp_module_weights(save_dict)
            save_dict = remove_quantizer_weights(save_dict)
            save_dict = convert_keys_to_awq_format(save_dict)

            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )
                
    # FIGNA PoC
    if args.use_custom_kernel:
        print("[FIGNA POC] Enable custom kernels for quantized layers")
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name in qlayers:
            print(f"Checking {name} for custom kernel support...")
            if hasattr(qlayers[name].module, 'int_weight'):
                print(f"Enabling custom kernel for {name} with int_weight shape {qlayers[name].module.int_weight.shape}")
                qlayers[name].use_custom_kernel = True
                print(f"Enable custom kernel for {name}")
            if hasattr(qlayers[name], "printed_once"):
                qlayers[name].printed_once = False
    
    # Custom Attention - Set flag on model config and wrappers
    if args.custom_attention:
        print("[FIGNA POC] Enable custom attention for K/V quantization")
        model.config.custom_attention = True
        
        # Set custom_attention flag on ActQuantWrappers (for V caching)
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name in qlayers:
            qlayers[name].custom_attention = True
            if 'v_proj' in name:
                print(f"  Enabled V caching for {name}")
        
        # Set custom_attention flag on QKRotationWrappers (for K caching)
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'apply_rotary_pos_emb_qk_rotation_wrapper'):
                layer.self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.custom_attention = True
                print(f"  Enabled K caching for layer")
                
    return model
