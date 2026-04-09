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

.PHONY: all clean test bench

all: bitnet_cli bitnet_bench

%.o: %.c include/bitnet.h
	$(CC) $(CFLAGS) -c $< -o $@

bitnet_cli: $(OBJS) tools/bitnet_cli.c
	$(CC) $(CFLAGS) -o $@ tools/bitnet_cli.c $(OBJS) $(LDFLAGS)

bitnet_bench: $(OBJS) tools/bitnet_bench.c
	$(CC) $(CFLAGS) -o $@ tools/bitnet_bench.c $(OBJS) $(LDFLAGS)

test_gguf: $(OBJS) tests/test_gguf.c
	$(CC) $(CFLAGS) -o $@ tests/test_gguf.c $(OBJS) $(LDFLAGS)

test_matmul: $(OBJS) tests/test_matmul.c
	$(CC) $(CFLAGS) -o $@ tests/test_matmul.c $(OBJS) $(LDFLAGS)

test_tokenizer: $(OBJS) tests/test_tokenizer.c
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer.c $(OBJS) $(LDFLAGS)

test_vs_reference: $(OBJS) tests/test_vs_reference.c
	$(CC) $(CFLAGS) -o $@ tests/test_vs_reference.c $(OBJS) $(LDFLAGS)

test: test_gguf test_matmul test_tokenizer
	@echo "=== Running all tests ==="
	./test_matmul
	@echo ""
	./test_gguf
	@echo ""
	./test_tokenizer

bench: bitnet_bench
	./bitnet_bench -m $(BITNET_MODEL) -t 4

clean:
	rm -f $(OBJS) bitnet_cli bitnet_bench test_gguf test_matmul test_tokenizer test_vs_reference
