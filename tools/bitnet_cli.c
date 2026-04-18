#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  -m <path>     Model GGUF file (required)\n"
        "  -p <text>     Prompt text (required)\n"
        "  -n <int>      Number of tokens to generate (default: 50)\n"
        "  -t <int>      Number of threads (default: 4)\n"
        "  --temperature <float>  Sampling temperature (default: 0.6)\n"
        "  --top-k <int>          Top-k filtering (default: 40)\n"
        "  --top-p <float>        Nucleus sampling (default: 0.9)\n"
        "  --seed <int>           Random seed (default: random)\n"
        "  --ctx <int>            Context size (default: 2048)\n"
        "\n", prog);
}

typedef struct {
    int tokens_generated;
} gen_state_t;

enum {
    CLI_MAX_PROMPT_TOKENS = 4096,
};

static bool on_token(int token, const char *text, void *ud) {
    gen_state_t *state = (gen_state_t *)ud;
    (void)token;
    printf("%s", text);
    fflush(stdout);
    state->tokens_generated++;
    return true;
}

static int parse_int_option(const char *opt, const char *value,
                            int min_value, const char *constraint,
                            int *out) {
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' ||
        parsed < min_value || parsed > INT_MAX) {
        fprintf(stderr, "Invalid value for %s: '%s' (%s)\n",
                opt, value, constraint);
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

static int parse_u64_option(const char *opt, const char *value,
                            uint64_t *out) {
    char *end = NULL;
    unsigned long long parsed;

    if (value[0] == '-') {
        fprintf(stderr,
                "Invalid value for %s: '%s' (must be an unsigned 64-bit integer)\n",
                opt, value);
        return 0;
    }

    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0') {
        fprintf(stderr,
                "Invalid value for %s: '%s' (must be an unsigned 64-bit integer)\n",
                opt, value);
        return 0;
    }

    *out = (uint64_t)parsed;
    return 1;
}

static int parse_float_option(const char *opt, const char *value,
                              const char *constraint, float *out) {
    char *end = NULL;
    float parsed;

    errno = 0;
    parsed = strtof(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' || !isfinite(parsed)) {
        fprintf(stderr, "Invalid value for %s: '%s' (%s)\n",
                opt, value, constraint);
        return 0;
    }

    *out = parsed;
    return 1;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    int n_predict = 50;
    bitnet_params_t params = bitnet_params_default();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parse_int_option("-n", argv[++i], 0,
                                  "must be a non-negative integer",
                                  &n_predict)) {
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            if (!parse_int_option("-t", argv[++i], 1,
                                  "must be a positive integer",
                                  &params.n_threads)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parse_float_option("--temperature", argv[++i],
                                    "must be a finite number",
                                    &params.temperature)) {
                return 1;
            }
            if (params.temperature < 0.0f) {
                fprintf(stderr,
                        "Invalid value for --temperature: '%s' (must be a non-negative number)\n",
                        argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_int_option("--top-k", argv[++i], 0,
                                  "must be a non-negative integer",
                                  &params.top_k)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            if (!parse_float_option("--top-p", argv[++i],
                                    "must be > 0 and <= 1",
                                    &params.top_p)) {
                return 1;
            }
            if (!(params.top_p > 0.0f && params.top_p <= 1.0f)) {
                fprintf(stderr,
                        "Invalid value for --top-p: '%s' (must be > 0 and <= 1)\n",
                        argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u64_option("--seed", argv[++i], &params.seed)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            if (!parse_int_option("--ctx", argv[++i], 1,
                                  "must be a positive integer",
                                  &params.n_ctx)) {
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !prompt) {
        fprintf(stderr, "Error: -m and -p are required\n");
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
    bitnet_model_t *model = bitnet_model_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        bitnet_model_free(model);
        return 1;
    }

    int tokens[CLI_MAX_PROMPT_TOKENS];
    int n_tokens = bitnet_tokenize(ctx, prompt, tokens, CLI_MAX_PROMPT_TOKENS);
    if (n_tokens <= 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        bitnet_ctx_free(ctx);
        bitnet_model_free(model);
        return 1;
    }
    if (n_tokens == CLI_MAX_PROMPT_TOKENS) {
        fprintf(stderr,
                "Error: prompt tokenization reached the CLI limit of %d tokens; prompt may be truncated\n",
                CLI_MAX_PROMPT_TOKENS);
        bitnet_ctx_free(ctx);
        bitnet_model_free(model);
        return 1;
    }
    if (n_tokens > params.n_ctx) {
        fprintf(stderr, "Error: prompt is %d tokens but --ctx is %d\n",
                n_tokens, params.n_ctx);
        bitnet_ctx_free(ctx);
        bitnet_model_free(model);
        return 1;
    }

    fprintf(stderr, "Prompt: %d tokens\n", n_tokens);

    printf("%s", prompt);
    fflush(stdout);

    gen_state_t state = {0};
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int n_gen = bitnet_generate(ctx, tokens, n_tokens, n_predict,
                                on_token, &state);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (double)(ts_end.tv_sec - ts_start.tv_sec) +
                     (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("\n");
    fprintf(stderr, "\n--- stats ---\n");
    fprintf(stderr, "Prompt tokens:    %d\n", n_tokens);
    fprintf(stderr, "Generated tokens: %d\n", n_gen);
    fprintf(stderr, "Total time:       %.2f s\n", elapsed);
    if (n_gen > 0) {
        fprintf(stderr, "Generation speed: %.2f tokens/sec\n",
                (double)n_gen / elapsed);
    }

    bitnet_ctx_free(ctx);
    bitnet_model_free(model);

    return 0;
}
