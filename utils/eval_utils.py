# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os

import torch
from tqdm import tqdm

from utils import model_utils

DEFAULT_ZERO_SHOT_TASKS = [
    "boolq",
    "piqa",
    "social_iqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]


@torch.no_grad()
def evaluator(model, testenc, dev, args):
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = (
        input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
    )  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
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

    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    position_ids = cache["position_ids"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]
    
    print("Evaluating the model layer by layer...")

    # for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
    for i in range(len(layers)):
        print(f"Processing layer {i+1}/{len(layers)}")
        layer = layers[i].to(dev)

        # Dump the layer input and output
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f"Dumped layer input and output to: {save_path}")

        for j in range(nbatches):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                #  defined.
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)

    model.lm_head = model.lm_head.to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")
    return ppl.item()


@torch.no_grad()
def zeroshot_evaluator(model, tokenizer, tasks=None, batch_size=4):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logging.warning(
            "lm-eval is not installed. Skipping zero-shot evaluation. "
            "Install with: pip install lm-eval"
        )
        return None

    if tasks is None:
        tasks = DEFAULT_ZERO_SHOT_TASKS

    print(f"Running zero-shot evaluation on: {tasks}", flush=True)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    import datasets as _hf_datasets
    _hf_datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        log_samples=False,
    )

    task_results = results["results"]

    # SpinQuant paper metric convention per task
    TASK_METRIC = {
        "boolq":         "acc,none",
        "piqa":          "acc_norm,none",
        "social_iqa":    "acc,none",
        "hellaswag":     "acc_norm,none",
        "winogrande":    "acc,none",
        "arc_easy":      "acc_norm,none",
        "arc_challenge": "acc_norm,none",
        "openbookqa":    "acc_norm,none",
    }

    print("\n========== Zero-shot Results ==========", flush=True)
    accs = []
    for task, metrics in task_results.items():
        preferred = TASK_METRIC.get(task)
        acc = metrics.get(preferred) if preferred else None
        # fallback: try acc_norm,none then acc,none
        if acc is None:
            acc = metrics.get("acc_norm,none")
        if acc is None:
            acc = metrics.get("acc,none")
        if acc is not None:
            accs.append(acc)
            print(f"  {task:20s}: {acc * 100:.2f}%  [{preferred or 'acc,none'}]", flush=True)

    if accs:
        avg = sum(accs) / len(accs)
        print(f"  {'Average':20s}: {avg * 100:.2f}%  ({len(accs)} tasks)", flush=True)
    print("=======================================\n", flush=True)

    return task_results
