#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
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

static void on_token(int token, const char *text, void *ud) {
    gen_state_t *state = (gen_state_t *)ud;
    (void)token;
    printf("%s", text);
    fflush(stdout);
    state->tokens_generated++;
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
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            params.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            params.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            params.seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
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

    int tokens[4096];
    int n_tokens = bitnet_tokenize(ctx, prompt, tokens, 4096);
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
