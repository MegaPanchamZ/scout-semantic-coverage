#!/usr/bin/env python3
"""Convert the released AutoVLA Lightning checkpoint into a merged HF directory.

See research/patches/README.md:
  AutoVLA_PDMS_89.ckpt (Lightning, fp32, `autovla.vlm.` prefix; vision keys under
  `model.visual.*`, LLM keys under `model.language_model.*`)
  -> checkpoints/AutoVLA-hf (bf16, 2 shards) loadable by
     Qwen2_5_VLForConditionalGeneration.from_pretrained(...)

The action tokenizer appends 2048 `<action_i>` tokens (ids 151665..153712) to the
Qwen2.5-VL tokenizer, so the merged embedding table must be resized to 153713.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch


def build_tokenizer(repo: str, qwen: str, action_start_id: int):
    from transformers import AutoProcessor

    sys.path.insert(0, repo)
    from models.action_tokenizer import ActionTokenizer

    processor = AutoProcessor.from_pretrained(qwen)
    tokenizer = processor.tokenizer
    before = len(tokenizer)
    at = ActionTokenizer(
        tokenizer,
        model_config={
            "tokens": {"action_start_id": action_start_id},
            "codebook_cache_path": os.path.join(repo, "codebook_cache", "agent_vocab.pkl"),
        },
    )
    print(f"[tokenizer] len before={before} after={len(tokenizer)} bins={at.n_bins}", flush=True)
    return processor, at


def extract_state_dict(ckpt_path: str) -> dict:
    print(f"[ckpt] loading {ckpt_path} (fp32, ~16GB) ...", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    print(f"[ckpt] raw keys={len(sd)}", flush=True)
    conv: dict = {}
    for k, v in sd.items():
        for p in ("autovla.vlm.", "vlm."):
            if k.startswith(p):
                conv[k[len(p):]] = v
                break
    if not conv:
        # fall back: strip everything up to the HF `model.` root
        raise SystemExit("no `autovla.vlm.`/`vlm.` keys found; inspect the checkpoint")
    print(f"[ckpt] converted keys={len(conv)}", flush=True)
    return conv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[1] / "models" / "AutoVLA"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--qwen", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-shard", default="5GB")
    ap.add_argument("--action-start-id", type=int, default=151665)
    args = ap.parse_args()

    args.ckpt = args.ckpt or os.path.join(args.repo, "checkpoints", "AutoVLA_PDMS_89.ckpt")
    args.qwen = args.qwen or os.path.join(args.repo, "Qwen2.5-VL-3B-Instruct")
    args.out = args.out or os.path.join(args.repo, "checkpoints", "AutoVLA-hf")

    processor, _ = build_tokenizer(args.repo, args.qwen, args.action_start_id)
    conv = extract_state_dict(args.ckpt)

    emb = conv.get("model.embed_tokens.weight")
    target_vocab = int(emb.shape[0]) if emb is not None else len(processor.tokenizer)
    print(f"[model] target vocab={target_vocab} (tokenizer len={len(processor.tokenizer)})", flush=True)

    from transformers import Qwen2_5_VLForConditionalGeneration

    print("[model] loading Qwen2.5-VL-3B base (bf16) ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if model.get_input_embeddings().weight.shape[0] != target_vocab:
        model.resize_token_embeddings(target_vocab)
        print(f"[model] resized embeddings -> {target_vocab}", flush=True)

    missing, unexpected = model.load_state_dict(conv, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("[load] missing sample:", missing[:8], flush=True)
    if unexpected:
        print("[load] unexpected sample:", unexpected[:8], flush=True)

    model = model.to(torch.bfloat16).eval()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[save] writing HF dir -> {args.out} (max_shard={args.max_shard})", flush=True)
    model.save_pretrained(args.out, max_shard_size=args.max_shard, safe_serialization=True)
    processor.save_pretrained(args.out)
    print("[save] done", flush=True)

    # sanity: reload and verify one tensor + vocab
    from transformers import AutoProcessor as AP, Qwen2_5_VLForConditionalGeneration as M

    m2 = M.from_pretrained(args.out, torch_dtype=torch.bfloat16)
    p2 = AP.from_pretrained(args.out)
    w1 = model.get_input_embeddings().weight[0, :4].float().tolist()
    w2 = m2.get_input_embeddings().weight[0, :4].float().tolist()
    print(f"[verify] embed[0,:4] saved={w2} matches={w1 == w2}", flush=True)
    print(f"[verify] reloaded vocab={m2.get_input_embeddings().weight.shape[0]} tokenizer_len={len(p2.tokenizer)}", flush=True)


if __name__ == "__main__":
    main()
