CC       ?= cc
CFLAGS   := -std=c11 -Wall -Wextra -Werror -pedantic -O2 \
            -D_POSIX_C_SOURCE=200809L -Iinclude
LDFLAGS  := -lm -lpthread

# SIMD selection: make SIMD=avx2 (or avx512, neon, scalar)
SIMD     ?= avx2

ifeq ($(SIMD),avx2)
    CFLAGS += -mavx2 -mfma
    MATMUL_SRC = src/bitnet_matmul_avx2.c src/bitnet_matmul_scalar.c
else ifeq ($(SIMD),neon)
    MATMUL_SRC = src/bitnet_matmul_scalar.c
else
    MATMUL_SRC = src/bitnet_matmul_scalar.c
endif

SRCS := src/bitnet_gguf.c \
        src/bitnet_arena.c \
        src/bitnet_quant.c \
        $(MATMUL_SRC) \
        src/bitnet_sampler.c \
        src/bitnet_tokenizer.c \
        src/bitnet_core.c

OBJS := $(SRCS:.c=.o)
OBJS_NO_CORE := $(filter-out src/bitnet_core.o,$(OBJS))

.PHONY: all clean test bench

all: bitnet_cli bitnet_bench bench_rope

%.o: %.c include/bitnet.h
	$(CC) $(CFLAGS) -c $< -o $@

bitnet_cli: $(OBJS) tools/bitnet_cli.c
	$(CC) $(CFLAGS) -o $@ tools/bitnet_cli.c $(OBJS) $(LDFLAGS)

bitnet_bench: $(OBJS) tools/bitnet_bench.c
	$(CC) $(CFLAGS) -o $@ tools/bitnet_bench.c $(OBJS) $(LDFLAGS)

bench_rope: tools/bench_rope.c
	$(CC) $(CFLAGS) -o $@ tools/bench_rope.c $(LDFLAGS)

test_gguf: $(OBJS) tests/test_gguf.c
	$(CC) $(CFLAGS) -o $@ tests/test_gguf.c $(OBJS) $(LDFLAGS)

test_matmul: $(OBJS) tests/test_matmul.c
	$(CC) $(CFLAGS) -o $@ tests/test_matmul.c $(OBJS) $(LDFLAGS)

test_thread_create_failures: $(OBJS_NO_CORE) tests/test_thread_create_failures.c
	$(CC) $(CFLAGS) -o $@ tests/test_thread_create_failures.c $(OBJS_NO_CORE) $(LDFLAGS)

test_quantizer: $(OBJS) tests/test_quantizer.c
	$(CC) $(CFLAGS) -o $@ tests/test_quantizer.c $(OBJS) $(LDFLAGS)

test_tokenizer: $(OBJS) tests/test_tokenizer.c
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer.c $(OBJS) $(LDFLAGS)

test_tokenizer_utf8: $(OBJS) tests/test_tokenizer_utf8.c
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer_utf8.c $(OBJS) $(LDFLAGS)

test_tokenizer_metadata: $(OBJS) tests/test_tokenizer_metadata.c
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer_metadata.c $(OBJS) $(LDFLAGS)

test_tokenizer_threads: $(OBJS) tests/test_tokenizer_threads.c
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer_threads.c $(OBJS) $(LDFLAGS)

test_arena: $(OBJS) tests/test_arena.c
	$(CC) $(CFLAGS) -o $@ tests/test_arena.c $(OBJS) $(LDFLAGS)

test_forward_guards: $(OBJS) tests/test_forward_guards.c
	$(CC) $(CFLAGS) -o $@ tests/test_forward_guards.c $(OBJS) $(LDFLAGS)

test_sampler_oom: $(OBJS) tests/test_sampler_oom.c
	$(CC) $(CFLAGS) -o $@ tests/test_sampler_oom.c $(OBJS) $(LDFLAGS)

test_sampler_init: $(OBJS) tests/test_sampler_init.c
	$(CC) $(CFLAGS) -o $@ tests/test_sampler_init.c $(OBJS) $(LDFLAGS)

test_vs_reference: $(OBJS) tests/test_vs_reference.c
	$(CC) $(CFLAGS) -o $@ tests/test_vs_reference.c $(OBJS) $(LDFLAGS)

test_rmsnorm: tests/test_rmsnorm.c
	$(CC) $(CFLAGS) -o $@ tests/test_rmsnorm.c $(LDFLAGS)

test_cli_args: tests/test_cli_args.c bitnet_cli bitnet_bench
	$(CC) $(CFLAGS) -o $@ tests/test_cli_args.c $(LDFLAGS)

test: test_gguf test_matmul test_thread_create_failures test_quantizer test_tokenizer_utf8 test_tokenizer_metadata test_tokenizer_threads test_arena test_forward_guards test_sampler_oom test_sampler_init test_rmsnorm test_cli_args
	@echo "=== Running all tests ==="
	./test_matmul
	@echo ""
	./test_thread_create_failures
	@echo ""
	./test_quantizer
	@echo ""
	./test_gguf
	@echo ""
	./test_tokenizer_utf8
	@echo ""
	./test_tokenizer_metadata
	@echo ""
	./test_tokenizer_threads
	@echo ""
	./test_arena
	@echo ""
	./test_forward_guards
	@echo ""
	./test_sampler_oom
	@echo ""
	./test_sampler_init
	@echo ""
	./test_rmsnorm
	@echo ""
	./test_cli_args

bench: bitnet_bench
	./bitnet_bench -m $(BITNET_MODEL) -t 4

clean:
	rm -f $(OBJS) bitnet_cli bitnet_bench bench_rope test_gguf test_matmul test_thread_create_failures test_quantizer test_tokenizer test_tokenizer_utf8 test_tokenizer_metadata test_tokenizer_threads test_arena test_forward_guards test_sampler_oom test_sampler_init test_vs_reference test_rmsnorm test_cli_args
