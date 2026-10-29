# Upstream CUDA performance review — 2026-10-28

Merging upstream v1.6.4 and v1.6.5 into `main` **does not measurably change
Gemma 4 E4B CUDA prefill or plain-decode rate** on an RTX PRO 4500 Blackwell.
Greedy output is token-identical. The upstream changes that do target speed
apply to other workloads: speculative decoding, quantized KV cache, recurrent
models, CPU-offloaded MoE, and models without post-norms.

## Merge scope

| Item | Revision / result |
| --- | --- |
| Previous fork tip | `879a7f2b` (upstream base `a3bfbfd0`, v1.6.3) |
| Upstream target | `cdc785db` (`upstream/master`, v1.6.5) |
| Incoming range | `a3bfbfd0..cdc785db`: 71 commits |
| Method | `git merge --no-ff` (no history rewrite) |
| Conflicts | `README.md` architecture table only: upstream table kept, fork's Gemma 4 E-series note kept |
| Semantic fix | `TransformerBlock`: the new deferred-residual path (`ca518f52`) returns before the fork's `post_mlp` hook runs. It now requires `post_mlp is None`. Gemma 4 never reached that path (it has `mlp_post_norm` and `layer_scalar`), so this is a guard, not a behavior change. |

## Validation

- Extension built from source for SM 12.0 (PyTorch 2.13.0+cu130).
- `tests/test_gemma4_e.py tests/test_gelu_precision.py tests/generator tests/model_setup tests/cache`: 296 passed, 47 skipped.
- `tests/kernels/attention tests/modules`: 888 passed, 15 skipped, 4 failed. All 4 are in `tests/modules/draft/test_dflash2.py` (`'LMHead' object has no attribute 'sparse_head'`) and fail identically on unmodified `upstream/master`.
- Gemma 4 12B Unified (`gemma-4-12B-it-qat-exl3-4.0bpw`) does not load on `main` before or after the merge: the released config lacks `vision_soft_tokens_per_image`. That fix exists only on the `research` branch (`12ede90a`).

## Measurement

E4B QAT EXL3 4.0 bpw, RTX PRO 4500 Blackwell (32 GB), batch 1, FP16 paged
cache 4096, max chunk 768, plain greedy decode of 128 tokens ignoring EOS,
prefix cache reset between runs, no FP16 prefill-weight cache, default
environment. One discarded warmup. Each cell is the range of medians over
three sessions (one of three runs and two of five runs). Prefill is prompt to
first token (includes the final-token forward and sampling).

| Build | Input tokens | Prefill, tok/s | First token, ms | Decode, tok/s |
| --- | ---: | ---: | ---: | ---: |
| `879a7f2b` (before) | 400 | 11,052–11,399 | 35.1–36.2 | 141.3–143.3 |
| merged | 400 | 10,738–11,223 | 35.6–37.3 | 141.8–142.8 |
| `879a7f2b` (before) | 930 | 12,895–13,589 | 68.4–72.1 | 141.9–143.4 |
| merged | 930 | 13,126–13,696 | 67.9–70.9 | 141.4–143.0 |

The ranges overlap; run-to-run spread (about ±3%) is larger than any
difference. This is a desktop GPU, not the Orin Nano numbers in the README;
Orin was not re-measured.

## Changes relevant to performance

- **Gemma 4 assistant drafter** (`e7af3420`, with `47d42fea` sparse-head MTP
  drafting and `288d0434` batched speculative readback). This is the largest
  potential decode gain for Gemma, but it does not run on this fork's targets
  yet. Upstream rejects `gemma-4-E4B-it-assistant` because it uses ordered
  (centroid) embeddings (`use_ordered_embeddings: true`, 2048 centroids). The
  12B drafter's architecture, `Gemma4UnifiedAssistantForCausalLM`, is not
  registered upstream. The drafter already reads `num_kv_shared_layers`, so
  E-series shared-KV targets are anticipated. Adding centroid-embedding support
  is the most promising follow-up for E4B decode.
- **Deferred residual add** (`ca518f52`). Folds a block's final residual add
  into the next block's input RMSNorm. It does not apply to Gemma 4, which has a
  post-MLP norm and a layer scalar. It helps Llama/Qwen-style blocks.
- **Single-call decode staging** (`51d1e799`) and cheaper `cuda_sync_active`
  (`c72d9ad1`). They reduce host overhead per generator step. At about 7 ms per
  E4B step on this GPU, no gain is visible. A slower host CPU (Orin) could
  benefit more; not measured.
- **Quantized-cache prefill staging** (`ab3e7b91`, `56ffdaea`). Caps the
  staging scratch (`EXL3_QC_STAGING_PAGES`) and uses 8-warp tiles to avoid
  register spill. This applies only with a quantized KV cache; the fork's E4B
  setup uses FP16.
- **Draft row ceiling** (`85cae3e7`) and **fair prefill/decode scheduling**
  (`b0d0d368`, `2ece09d1`, off by default). They affect batching and latency
  under mixed load, not single-stream kernel rate.
- **GDN b/a GEMV over 16 rows** (`d33d1869`). Recurrent models only.
- **CPU-MoE paired AVX2/AVX-512 kernels and load fixes** (`d4117b7b`,
  `10617057`) and conversion changes. They do not apply to fully GPU-resident
  dense inference.

## Reproduce

```sh
git log --oneline a3bfbfd0..cdc785db
python -m pytest tests/test_gemma4_e.py tests/test_gelu_precision.py \
  tests/generator tests/model_setup tests/cache -q
```
