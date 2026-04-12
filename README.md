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
- **Multi-threaded** &mdash; pthread-based parallel matmul for both I2_S and F32 layers
- **Arena allocator** &mdash; zero allocations during inference after init
- **Thread-safe** &mdash; no global state, all mutable data lives in the context

## Getting a Model

Download the official BitNet b1.58 2B-4T model in GGUF format:

```sh
# Using huggingface-cli
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --local-dir models/

# Or manually from:
# https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf
```

The model file you need is `ggml-model-i2_s.gguf`.

## Build

```sh
make                # AVX2 (default, recommended for x86-64)
make SIMD=scalar    # portable scalar fallback (any platform)
```

Requires a C11 compiler (gcc, clang) and pthreads. That's it.

## Usage

```sh
./bitnet_cli -m models/ggml-model-i2_s.gguf -p "The meaning of life is" -n 50 -t 4
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

Unit tests cover the GGUF parser, matmul kernels (scalar + AVX2), and tokenizer.
They require a model file &mdash; set `BITNET_MODEL` to point at it:

```sh
export BITNET_MODEL=/path/to/ggml-model-i2_s.gguf

make test               # GGUF parser, matmul kernels, tokenizer
make test_vs_reference  # compare output against llama.cpp (optional)
```

The matmul test (`test_matmul`) is self-contained and does not need a model file.

### Reference Comparison

`test_vs_reference` compares bitnet-c11 against Microsoft's llama.cpp-based implementation.
It verifies tokenizer equivalence, output coherence, performance, and determinism.
Requires `llama-cli` and `llama-tokenize` on `$PATH` (or set `LLAMA_CLI` / `LLAMA_TOKENIZE`):

```sh
export LLAMA_CLI=/path/to/llama-cli
export LLAMA_TOKENIZE=/path/to/llama-tokenize
./test_vs_reference
```

## Benchmark

```sh
make bench BITNET_MODEL=/path/to/ggml-model-i2_s.gguf
```

### llama.cpp Comparison

`make compare` runs a consolidated benchmark against llama.cpp and outputs
structured JSON with prompt eval and generation metrics for both engines:

```sh
export BITNET_MODEL=/path/to/ggml-model-i2_s.gguf
export LLAMA_CLI=/path/to/llama-cli       # or ensure llama-cli is on $PATH
make compare
```

The tool exits non-zero if the model file is missing or `llama-cli` is not found.
JSON output goes to stdout; progress and a human-readable summary go to stderr.
Redirect stdout to capture the artifact:

```sh
make compare > comparison.json
```

## Performance

On a 4-core Xeon E-2224G with AVX2 (single-socket, no GPU):

| Metric | bitnet-c11 | llama.cpp (reference) |
|--------|------------|----------------------|
| Generation | 7.7 tok/s | 6.9 tok/s |
| Prompt eval | ~15 tok/s | ~12 tok/s |

bitnet-c11 is competitive with (and slightly faster than) the reference C++
implementation for this model, likely due to lower framework overhead.

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
  bitnet_sampler.c          Temperature / top-k / top-p sampling
  bitnet_core.c             Model loading, transformer forward pass

tools/
  bitnet_cli.c              Command-line inference tool
  bitnet_bench.c            Benchmark (prompt + generation speed)
  compare_llama.c           Consolidated comparison runner vs llama.cpp

tests/
  test_gguf.c               GGUF parser correctness
  test_matmul.c             Matmul kernel correctness (scalar vs AVX2)
  test_tokenizer.c          Tokenizer encode/decode round-trips
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
for the unsigned-times-signed multiply, achieving near-peak throughput.

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

// Cleanup
bitnet_ctx_free(ctx);
bitnet_model_free(model);
```

## Supported Models

| Model | Parameters | GGUF Type | Status |
|-------|-----------|-----------|--------|
| BitNet-b1.58-2B-4T | 2B | I2_S | Tested |

