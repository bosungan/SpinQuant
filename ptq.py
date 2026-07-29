# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()
    
    model = ptq_model(ptq_args, model, model_args)
    # GPTQ/rotation leave weights on CPU (layer-by-layer .cpu()), while buffers
    # like rotary_emb.inv_freq stay on CUDA -> device mismatch during zero-shot.
    # We must move the whole model to CUDA, but each quantized Linear carries a
    # redundant representation: fp16 `weight` (used by the standard path) AND
    # int8 `int_weight` + a FULL-SIZE fp32 `scale` buffer (used only by the
    # custom FIGNA kernel). Together that is ~48GB for 8B and .cuda() OOMs.
    # Drop whichever representation this run does not use before moving to GPU.
    import gc
    _use_ck = ptq_args.use_custom_kernel
    for _m in model.modules():
        if "int_weight" in getattr(_m, "_buffers", {}):
            if _use_ck:
                # custom kernel uses int_weight+scale; free the unused fp16 weight
                # and downcast the FULL-SIZE fp32 `scale` buffer to fp16 (the
                # custom GEMM upcasts to fp32 internally, so this is lossless for
                # the kernel) to halve its ~28GB footprint on the 8B model.
                if getattr(_m, "weight", None) is not None:
                    _m._parameters.pop("weight", None)
                _sc = _m._buffers.get("scale", None)
                if _sc is not None and _sc.dtype == torch.float32:
                    _m._buffers["scale"] = _sc.half()
            else:
                # standard path uses fp16 weight; free unused int_weight/scale
                for _b in ("int_weight", "scale", "groupsize"):
                    _m._buffers.pop(_b, None)
    gc.collect()
    torch.cuda.empty_cache()
    model.cuda()
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    if ptq_args.eval_ppl:
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        log.info("wiki2 ppl is: {}".format(dataset_ppl))

    if ptq_args.eval_zero_shot:
        tasks = [t.strip() for t in ptq_args.zero_shot_tasks.split(",")]
        eval_utils.zeroshot_evaluator(model, tokenizer, tasks, batch_size=ptq_args.bsz)

    dist.barrier()


if __name__ == "__main__":
    train()
