# bitnet-c11

Pure C11 inference library for 1-bit LLMs (BitNet b1.58).

No C++. No Python. No external dependencies beyond libc, libm, and pthreads.

## Features

- **Strict C11** &mdash; compiles with `-std=c11 -Wall -Wextra -Werror -pedantic`
- **Zero dependencies** &mdash; just libc, libm, pthreads. No cmake, no autotools
- **GGUF V3 parser** &mdash; mmap-based, zero-copy weight access
- **I2_S quantization** &mdash; native 2-bit ternary weight unpacking
- **GPT-2 BPE tokenizer** &mdash; byte-level encoding, reads vocab/merges from GGUF metadata
- **AVX2 SIMD kernel** &mdash; `_mm256_maddubs_epi16` for ternary GEMV, FMA for F32 output projection
- **AVX-512 VNNI kernel** (opt-in) &mdash; `_mm512_dpbusd_epi32` variant, built via `make SIMD=avx512`
- **Multi-threaded** &mdash; atomic spin-based thread pool (`bn_pool_create`) for parallel I2_S GEMV (`bn_gemv_mt`) and F32 matmul (`bn_matmul_f32`), with ~1 &mu;s dispatch latency
- **Arena allocator** &mdash; zero allocations during inference after init
- **Thread-safe** &mdash; no global state, all mutable data lives in the context
- **Attention timing** &mdash; `bitnet_attn_time_reset()` exposes accumulated attention time for profiling

## Getting a Model

Download the official BitNet b1.58 2B-4T model in GGUF format:

```sh
# Using huggingface-cli
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --local-dir models/

# Or manually from:
# https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf
```

The model file is `ggml-model-i2_s.gguf` inside the downloaded directory
(e.g. `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`).

## Build

```sh
make                # AVX2 (default, recommended for x86-64)
make SIMD=avx512    # AVX-512 + VNNI (auto-dispatches at runtime when built)
make SIMD=scalar    # portable scalar fallback (any platform)
```

The AVX-512 build requires a CPU with AVX-512F, AVX-512BW, AVX-512VL, and
AVX-512VNNI. It produces byte-identical output to the AVX2 build; on most
Xeon / EPYC / Ice Lake-class CPUs the two are within noise, since generation
is not bound by the GEMV kernel at typical BitNet-2B sizes.

Requires a C11 compiler (gcc, clang) and pthreads. That's it.

Build targets:

| Target | Output | Description |
|--------|--------|-------------|
| `make` | `bitnet_cli`, `bitnet_bench`, `bench_rope` | Default: CLI, benchmark, and RoPE bench |
| `make test` | 13 test binaries | Build and run the full test suite |
| `make bench` | `bitnet_bench` | Full-model + micro benchmark tool |
| `make compare` | `compare_llama` | Side-by-side comparison vs llama.cpp |

## Usage

```sh
./bitnet_cli -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "The meaning of life is" -n 50 -t 4
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-m <path>` | GGUF model file (required) | &mdash; |
| `-p <text>` | Prompt text (required) | &mdash; |
| `-n <int>` | Tokens to generate | 50 |
| `-t <int>` | Threads | 4 |
| `--temperature <float>` | Sampling temperature (0 = greedy) | 0.6 |
| `--top-k <int>` | Top-k filtering | 40 |
| `--top-p <float>` | Nucleus sampling threshold | 0.9 |
| `--seed <int>` | RNG seed (deterministic output) | random |
| `--ctx <int>` | Context window size | 2048 |

## Tests

The test suite covers the GGUF parser, matmul kernels, quantizer, tokenizer
(including UTF-8 edge cases, metadata, and thread safety), arena allocator,
forward-pass guards, sampler edge cases, RMS normalization, and CLI argument
parsing. Most tests require a model file &mdash; set `BITNET_MODEL` to point
at it:

```sh
export BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

make test               # build and run all 13 tests
make test_vs_reference  # compare output against llama.cpp (optional)
```

### Test Matrix

`make test` builds and runs the following:

| Test | What it covers |
|------|----------------|
| `test_matmul` | Matmul kernel correctness (scalar vs AVX2) |
| `test_thread_create_failures` | Thread pool resilience under creation failures |
| `test_quantizer` | Activation quantization round-trips |
| `test_gguf` | GGUF V3 parser correctness |
| `test_tokenizer_utf8` | Tokenizer UTF-8 encoding edge cases |
| `test_tokenizer_metadata` | Tokenizer metadata extraction from GGUF |
| `test_tokenizer_threads` | Concurrent tokenizer access |
| `test_arena` | Arena allocator behavior |
| `test_forward_guards` | Forward-pass argument validation |
| `test_sampler_oom` | Sampler behavior under allocation failure |
| `test_sampler_init` | Sampler initialization and defaults |
| `test_rmsnorm` | RMS normalization correctness |
| `test_cli_args` | CLI and bench argument parsing |

`test_matmul` and `test_rmsnorm` are self-contained and do not need a model
file. `test_cli_args` requires `bitnet_cli` and `bitnet_bench` to be built
first (handled automatically by `make test`).

### Reference Comparison

`test_vs_reference` compares bitnet-c11 against Microsoft's llama.cpp-based implementation.
It verifies tokenizer equivalence, output coherence, performance, and determinism.
Requires `llama-cli` and `llama-tokenize` on `$PATH` (or set `LLAMA_CLI` / `LLAMA_TOKENIZE`).

## Benchmark

`bitnet_bench` supports two modes:

**Full-model benchmark** &mdash; measures prompt processing and token generation
throughput on a real model, reports attention time breakdown via
`bitnet_attn_time_reset()`, and emits structured JSON to stdout:

```sh
make bench BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# or run directly:
./bitnet_bench -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 4
```

The JSON output includes `pp_tok_s` (prompt eval tokens/sec), `tg_tok_s`
(generation tokens/sec), and `pp_attn_ms` / `tg_attn_ms` (attention time in
milliseconds for each phase).

**Kernel microbenchmark** (`--micro`) &mdash; benchmarks I2_S GEMV and F32
matmul kernels at representative layer shapes, reporting single-thread and
multi-thread latency with speedup ratios:

```sh
./bitnet_bench --micro -t 4
```

### llama.cpp Comparison

`make compare` benchmarks bitnet-c11 and llama.cpp side-by-side on the same
prompt, thread count, and generation length, then emits structured JSON to
stdout and a human-readable summary to stderr:

```sh
export BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
export LLAMA_CLI=/path/to/llama-cli   # or ensure llama-cli is on $PATH
make compare                          # summary on stderr, JSON on stdout
make compare > comparison.json        # save the JSON artifact
```

The JSON output includes per-engine prompt eval and generation throughput, plus
an overall generation speedup ratio. The runner exits non-zero if the model
file or `llama-cli` is missing. See `tools/compare_llama.c` for the full
methodology: greedy decoding, 32-token generation, 4 threads,
`CLOCK_MONOTONIC` timing.

## Performance

Measured with `make compare` on an 8-core Intel Xeon @ 2.8 GHz (AVX-512 VNNI
available, single-socket, no GPU), BitNet-b1.58-2B-4T model, greedy decoding.
`compare_llama` hard-codes 4 threads (`N_THREADS 4`) and 32 generated tokens,
so the numbers below reflect 4-thread execution:

| Metric | bitnet-c11 (AVX2 build) | llama.cpp (AVX-VNNI) | Ratio |
|--------|-------------------------|----------------------|-------|
| Prompt eval | 31.2 tok/s | 20.1 tok/s | **1.55&times; bitnet-c11** |
| Generation | 17.6 tok/s | 20.6 tok/s | 0.85&times; (llama.cpp 1.17&times; faster) |

The AVX-512 build produces byte-identical output to the AVX2 build and
performs within noise on the same hardware (~15.5 vs ~15.7 tok/s generation
at 4 threads); the GEMV kernel is not the bottleneck at 2B parameters, so
VNNI's reduced op count doesn't translate into end-to-end speedup. The
remaining gap to llama.cpp is elsewhere in the per-layer pipeline.

To reproduce, run `make compare` with `BITNET_MODEL` pointing at
`models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` and `LLAMA_CLI` pointing
at a llama.cpp build. Full JSON results go to stdout; redirect to a file to
keep them. Adjust the thread count by editing `N_THREADS` in
`tools/compare_llama.c`, or use `bitnet_bench -t <N>` for standalone
bitnet-c11 benchmarks at different thread counts.

## Architecture

```
include/
  bitnet.h                  Public API (single header)

src/
  bitnet_gguf.c             GGUF V3 parser (mmap-based, read-only)
  bitnet_arena.c            Arena allocator (64-byte SIMD alignment)
  bitnet_tokenizer.c        GPT-2 byte-level BPE tokenizer
  bitnet_quant.c            Activation quantization (float -> int8)
  bitnet_matmul_scalar.c    Portable scalar I2_S GEMV kernel
  bitnet_matmul_avx2.c      AVX2 optimized I2_S GEMV kernel
  bitnet_matmul_avx512.c    AVX-512 + VNNI I2_S GEMV kernel (opt-in via SIMD=avx512)
  bitnet_sampler.c          Temperature / top-k / top-p sampling
  bitnet_core.c             Model loading, atomic spin thread pool, transformer forward pass

tools/
  bitnet_cli.c              Command-line inference tool
  bitnet_bench.c            Benchmark (full-model + kernel microbenchmarks)
  compare_llama.c           Consolidated comparison runner vs llama.cpp (JSON + summary)
  bench_rope.c              RoPE kernel benchmark

tests/
  test_gguf.c               GGUF parser correctness
  test_matmul.c             Matmul kernel correctness (scalar vs AVX2)
  test_thread_create_failures.c  Thread pool creation failure handling
  test_quantizer.c          Activation quantization round-trips
  test_tokenizer.c          Tokenizer encode/decode round-trips
  test_tokenizer_utf8.c     Tokenizer UTF-8 edge cases
  test_tokenizer_metadata.c Tokenizer metadata extraction
  test_tokenizer_threads.c  Concurrent tokenizer access
  test_arena.c              Arena allocator behavior
  test_forward_guards.c     Forward-pass argument validation
  test_sampler_oom.c        Sampler OOM resilience
  test_sampler_init.c       Sampler initialization
  test_rmsnorm.c            RMS normalization correctness
  test_cli_args.c           CLI argument parsing
  test_vs_reference.c       Full comparison against llama.cpp
```

### Model Support

Reads GGUF V3 files with I2_S quantization as produced by
[Microsoft's bitnet.cpp](https://github.com/microsoft/BitNet). Tested with:

- **BitNet-b1.58-2B-4T** &mdash; 2 billion parameters, 30 layers, 2560 embedding dim

The model uses tied embeddings (token_embd in F16, output projection reuses it),
grouped-query attention (20 Q heads / 5 KV heads), squared ReLU activation,
and RoPE positional encoding.

### BitLinear

The core operation for ternary weight layers:

1. Quantize activations: `x_int8 = round(x * 127 / max|x|)`
2. Ternary GEMV: `raw = W_i2s @ x_int8` (weights are {-1, 0, +1})
3. Dequantize: `output = (raw - sum(x_int8)) * weight_scale / act_scale`

The AVX2 kernel processes 128 elements per block using `_mm256_maddubs_epi16`
for the unsigned-times-signed multiply. The optional AVX-512 kernel processes
256 elements per block using `_mm512_dpbusd_epi32` (VNNI), accumulating
directly into int32 without the 16-bit intermediate.

## Library API

```c
#include "bitnet.h"

// Load model (mmap, zero-copy)
bitnet_model_t *model = bitnet_model_load("model.gguf");

// Create inference context
bitnet_params_t params = bitnet_params_default();
params.n_threads = 4;
bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);

// Tokenize
int tokens[256];
int n = bitnet_tokenize(ctx, "Hello, world", tokens, 256);

// Generate with callback
bitnet_generate(ctx, tokens, n, 100, my_callback, NULL);

// Or step-by-step: forward pass + sample
float *logits = bitnet_forward(ctx, tokens, n, true);
int next = bitnet_sample_token(ctx, logits);

// Attention profiling
double attn_sec = bitnet_attn_time_reset(ctx);

// Cleanup
bitnet_ctx_free(ctx);
bitnet_model_free(model);
```

## Supported Models

| Model | Parameters | GGUF Type | Status |
|-------|-----------|-----------|--------|
| BitNet-b1.58-2B-4T | 2B | I2_S | Tested |
