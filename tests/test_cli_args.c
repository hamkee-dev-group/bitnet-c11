#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct {
    const char *name;
    uint32_t type;
    uint32_t n_dims;
    uint64_t ne[2];
} fixture_tensor_t;

typedef struct {
    uint32_t tokens_elem_type;
    uint32_t merges_elem_type;
    int include_bos;
    uint32_t bos_type;
    uint32_t eos_type;
} bench_fixture_config_t;

static void read_file(const char *path, char *buf, size_t size) {
    FILE *fp = fopen(path, "r");
    size_t nread;

    assert(fp != NULL);
    nread = fread(buf, 1, size - 1, fp);
    buf[nread] = '\0';
    fclose(fp);
}

static void write_u32(FILE *fp, uint32_t value) {
    assert(fwrite(&value, sizeof(value), 1, fp) == 1);
}

static void write_u64(FILE *fp, uint64_t value) {
    assert(fwrite(&value, sizeof(value), 1, fp) == 1);
}

static void write_str(FILE *fp, const char *value) {
    uint64_t len = strlen(value);
    write_u64(fp, len);
    assert(fwrite(value, 1, len, fp) == len);
}

static void write_padding(FILE *fp, size_t alignment) {
    long pos = ftell(fp);
    size_t pad;
    static const unsigned char zeros[32] = {0};

    assert(pos >= 0);
    pad = (alignment - ((size_t)pos % alignment)) % alignment;
    assert(pad <= sizeof(zeros));
    if (pad > 0) assert(fwrite(zeros, 1, pad, fp) == pad);
}

static void write_fixture_scalar(FILE *fp, uint32_t type, uint32_t u32_value,
                                 const char *str_value) {
    write_u32(fp, type);
    switch (type) {
    case BN_GGUF_TYPE_UINT32:
        write_u32(fp, u32_value);
        break;
    case BN_GGUF_TYPE_STRING:
        write_str(fp, str_value);
        break;
    default:
        assert(!"unsupported fixture scalar type");
    }
}

static size_t fixture_tensor_nbytes(const fixture_tensor_t *t) {
    size_t n_elem = 1;

    for (uint32_t i = 0; i < t->n_dims; i++) {
        n_elem *= (size_t)t->ne[i];
    }

    switch (t->type) {
    case BN_GGML_TYPE_F32:
        return n_elem * sizeof(float);
    case BN_GGML_TYPE_I2_S:
        return (n_elem / 4) + sizeof(float);
    default:
        assert(!"unsupported fixture tensor type");
        return 0;
    }
}

static void write_fixture_tensor_data(FILE *fp, const fixture_tensor_t *t) {
    size_t nbytes = fixture_tensor_nbytes(t);

    if (t->type == BN_GGML_TYPE_F32) {
        size_t n_elem = nbytes / sizeof(float);
        for (size_t i = 0; i < n_elem; i++) {
            float value = 1.0f + (float)i;
            assert(fwrite(&value, sizeof(value), 1, fp) == 1);
        }
        return;
    }

    for (size_t i = 0; i < nbytes - sizeof(float); i++) {
        uint8_t value = (uint8_t)(i & 0xffu);
        assert(fwrite(&value, sizeof(value), 1, fp) == 1);
    }

    {
        float weight_scale = 1.0f;
        assert(fwrite(&weight_scale, sizeof(weight_scale), 1, fp) == 1);
    }
}

static uint64_t align_up(uint64_t val, uint64_t alignment) {
    return ((val + alignment - 1) / alignment) * alignment;
}

static void write_zero_padding_to(FILE *fp, uint64_t target_offset,
                                   uint64_t data_section_start) {
    long cur = ftell(fp);
    assert(cur >= 0);
    uint64_t cur_data_rel = (uint64_t)cur - data_section_start;
    if (cur_data_rel < target_offset) {
        uint64_t pad = target_offset - cur_data_rel;
        static const unsigned char zeros[64] = {0};
        while (pad > 0) {
            uint64_t chunk = pad < sizeof(zeros) ? pad : sizeof(zeros);
            assert(fwrite(zeros, 1, (size_t)chunk, fp) == (size_t)chunk);
            pad -= chunk;
        }
    }
}

static void create_tiny_bench_fixture(char path_template[],
                                      const bench_fixture_config_t *config) {
    enum {
        FIXTURE_N_VOCAB = 256,
        FIXTURE_N_EMBD = 128,
        FIXTURE_N_FF = 128,
    };
    static const fixture_tensor_t tensors[] = {
        { "token_embd.weight",          BN_GGML_TYPE_F32, 2, { FIXTURE_N_EMBD, FIXTURE_N_VOCAB } },
        { "output_norm.weight",         BN_GGML_TYPE_F32, 1, { FIXTURE_N_EMBD, 1 } },
        { "blk.0.attn_norm.weight",     BN_GGML_TYPE_F32, 1, { FIXTURE_N_EMBD, 1 } },
        { "blk.0.attn_sub_norm.weight", BN_GGML_TYPE_F32, 1, { FIXTURE_N_EMBD, 1 } },
        { "blk.0.ffn_norm.weight",      BN_GGML_TYPE_F32, 1, { FIXTURE_N_EMBD, 1 } },
        { "blk.0.ffn_sub_norm.weight",  BN_GGML_TYPE_F32, 1, { FIXTURE_N_EMBD, 1 } },
        { "blk.0.attn_q.weight",        BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_EMBD, FIXTURE_N_EMBD } },
        { "blk.0.attn_q.weight.scale",  BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_k.weight",        BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_EMBD, FIXTURE_N_EMBD } },
        { "blk.0.attn_k.weight.scale",  BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_v.weight",        BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_EMBD, FIXTURE_N_EMBD } },
        { "blk.0.attn_v.weight.scale",  BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_output.weight",   BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_EMBD, FIXTURE_N_EMBD } },
        { "blk.0.attn_output.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_gate.weight",      BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_FF, FIXTURE_N_EMBD } },
        { "blk.0.ffn_gate.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_up.weight",        BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_FF, FIXTURE_N_EMBD } },
        { "blk.0.ffn_up.weight.scale",  BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_down.weight",      BN_GGML_TYPE_I2_S, 2, { FIXTURE_N_EMBD, FIXTURE_N_FF } },
        { "blk.0.ffn_down.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
    };
    const uint64_t n_tensors = sizeof(tensors) / sizeof(tensors[0]);
    uint64_t offset = 0;
    int fd = mkstemp(path_template);
    FILE *fp;

    assert(fd >= 0);
    fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, n_tensors);
    write_u64(fp, config->include_bos ? 15 : 14);

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "bitnet-b1.58");

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "bitnet-b1.58.vocab_size");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, FIXTURE_N_VOCAB);

    write_str(fp, "bitnet-b1.58.embedding_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, FIXTURE_N_EMBD);

    write_str(fp, "bitnet-b1.58.block_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count_kv");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.feed_forward_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, FIXTURE_N_FF);

    write_str(fp, "bitnet-b1.58.context_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 8192);

    write_str(fp, "bitnet-b1.58.attention.layer_norm_rms_epsilon");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float eps = 1e-5f;
        assert(fwrite(&eps, sizeof(eps), 1, fp) == 1);
    }

    write_str(fp, "bitnet-b1.58.rope.freq_base");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float freq_base = 10000.0f;
        assert(fwrite(&freq_base, sizeof(freq_base), 1, fp) == 1);
    }

    write_str(fp, "tokenizer.ggml.tokens");
    write_u32(fp, BN_GGUF_TYPE_ARRAY);
    write_u32(fp, config->tokens_elem_type);
    write_u64(fp, FIXTURE_N_VOCAB);
    if (config->tokens_elem_type == BN_GGUF_TYPE_STRING) {
        for (int i = 0; i < FIXTURE_N_VOCAB; i++) {
            char token[16];
            snprintf(token, sizeof(token), "<0x%02X>", i);
            write_str(fp, token);
        }
    } else if (config->tokens_elem_type == BN_GGUF_TYPE_UINT32) {
        for (int i = 0; i < FIXTURE_N_VOCAB; i++) {
            write_u32(fp, (uint32_t)i);
        }
    } else {
        assert(!"unsupported tokenizer token fixture element type");
    }

    write_str(fp, "tokenizer.ggml.merges");
    write_u32(fp, BN_GGUF_TYPE_ARRAY);
    write_u32(fp, config->merges_elem_type);
    write_u64(fp, 1);
    if (config->merges_elem_type == BN_GGUF_TYPE_STRING) {
        write_str(fp, "<0x20> <0x54>");
    } else if (config->merges_elem_type == BN_GGUF_TYPE_UINT32) {
        write_u32(fp, 0);
    } else {
        assert(!"unsupported tokenizer merge fixture element type");
    }

    if (config->include_bos) {
        write_str(fp, "tokenizer.ggml.bos_token_id");
        write_fixture_scalar(fp, config->bos_type, 0, "bos");
    }

    write_str(fp, "tokenizer.ggml.eos_token_id");
    write_fixture_scalar(fp, config->eos_type, 1, "eos");

    uint64_t offsets[sizeof(tensors)/sizeof(tensors[0])];
    for (uint64_t i = 0; i < n_tensors; i++) {
        const fixture_tensor_t *t = &tensors[i];

        offsets[i] = offset;
        write_str(fp, t->name);
        write_u32(fp, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            write_u64(fp, t->ne[d]);
        }
        write_u32(fp, t->type);
        write_u64(fp, offset);
        offset += fixture_tensor_nbytes(t);
        offset = align_up(offset, 32);
    }

    write_padding(fp, 32);
    uint64_t data_start = (uint64_t)ftell(fp);

    for (uint64_t i = 0; i < n_tensors; i++) {
        write_zero_padding_to(fp, offsets[i], data_start);
        write_fixture_tensor_data(fp, &tensors[i]);
    }

    assert(fclose(fp) == 0);
}

static void run_case(const char *label, const char *command,
                     const char *expected_error) {
    char stderr_path[] = "/tmp/bitnet-cli-test-XXXXXX";
    char cmd[512];
    char stderr_buf[2048];
    int fd;
    int status;

    fd = mkstemp(stderr_path);
    assert(fd >= 0);
    close(fd);

    snprintf(cmd, sizeof(cmd), "%s >/dev/null 2>%s", command, stderr_path);
    status = system(cmd);
    assert(status != -1);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) != 0);

    read_file(stderr_path, stderr_buf, sizeof(stderr_buf));
    if (strstr(stderr_buf, expected_error) == NULL) {
        fprintf(stderr, "%s\nExpected stderr to contain: %s\nActual stderr: %s\n",
                label, expected_error, stderr_buf);
        assert(0);
    }

    unlink(stderr_path);
    printf("%s: OK\n", label);
}

static void run_argv_case(const char *label, const char *const argv[],
                          const char *expected_error) {
    char stderr_path[] = "/tmp/bitnet-cli-test-XXXXXX";
    char stderr_buf[4096];
    int fd;
    int status;
    pid_t pid;

    fd = mkstemp(stderr_path);
    assert(fd >= 0);
    close(fd);

    pid = fork();
    assert(pid >= 0);

    if (pid == 0) {
        assert(freopen("/dev/null", "w", stdout) != NULL);
        assert(freopen(stderr_path, "w", stderr) != NULL);
        execv(argv[0], (char *const *)argv);
        perror("execv");
        _exit(127);
    }

    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) != 0);

    read_file(stderr_path, stderr_buf, sizeof(stderr_buf));
    if (strstr(stderr_buf, expected_error) == NULL) {
        fprintf(stderr, "%s\nExpected stderr to contain: %s\nActual stderr: %s\n",
                label, expected_error, stderr_buf);
        assert(0);
    }

    unlink(stderr_path);
    printf("%s: OK\n", label);
}

static void run_bench_fixture_error_case(const char *label, const char *expected_error) {
    char model_path[] = "/tmp/bitnet-bench-fixture-XXXXXX";
    char stderr_path[] = "/tmp/bitnet-bench-stderr-XXXXXX";
    char cmd[1024];
    char stderr_buf[4096];
    int fd;
    int status;
    static const bench_fixture_config_t config = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        0,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };

    create_tiny_bench_fixture(model_path, &config);

    fd = mkstemp(stderr_path);
    assert(fd >= 0);
    close(fd);

    snprintf(cmd, sizeof(cmd), "./bitnet_bench -m %s >/dev/null 2>%s",
             model_path, stderr_path);
    status = system(cmd);
    assert(status != -1);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) != 0);

    read_file(stderr_path, stderr_buf, sizeof(stderr_buf));
    if (strstr(stderr_buf, expected_error) == NULL) {
        fprintf(stderr, "%s\nExpected stderr to contain: %s\nActual stderr: %s\n",
                label, expected_error, stderr_buf);
        assert(0);
    }

    unlink(stderr_path);
    unlink(model_path);
    printf("%s: OK\n", label);
}

static void run_bench_fixture_success_case(const char *label) {
    char model_path[] = "/tmp/bitnet-bench-fixture-XXXXXX";
    char stderr_path[] = "/tmp/bitnet-bench-stderr-XXXXXX";
    char stdout_path[] = "/tmp/bitnet-bench-stdout-XXXXXX";
    char cmd[1024];
    char stderr_buf[4096];
    char stdout_buf[4096];
    int fd;
    int status;
    static const bench_fixture_config_t config = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        1,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };

    create_tiny_bench_fixture(model_path, &config);

    fd = mkstemp(stderr_path);
    assert(fd >= 0);
    close(fd);

    fd = mkstemp(stdout_path);
    assert(fd >= 0);
    close(fd);

    snprintf(cmd, sizeof(cmd), "./bitnet_bench -m %s >%s 2>%s",
             model_path, stdout_path, stderr_path);
    status = system(cmd);
    assert(status != -1);
    assert(WIFEXITED(status));

    read_file(stderr_path, stderr_buf, sizeof(stderr_buf));
    if (WEXITSTATUS(status) != 0) {
        fprintf(stderr,
                "%s\nExpected benchmark success.\nExit status: %d\nModel: %s\nActual stderr: %s\n",
                label, WEXITSTATUS(status), model_path, stderr_buf);
        fflush(stderr);
        assert(0);
    }
    if (strstr(stderr_buf, "--- Token Generation Benchmark ---") == NULL) {
        fprintf(stderr, "%s\nExpected benchmark to reach generation.\nActual stderr: %s\n",
                label, stderr_buf);
        assert(0);
    }

    read_file(stdout_path, stdout_buf, sizeof(stdout_buf));

    static const char *required_fields[] = {
        "\"pp_tokens\":", "\"pp_tok_s\":",
        "\"tg_tokens\":", "\"tg_tok_s\":",
        "\"pp_attn_ms\":", "\"tg_attn_ms\":",
        "\"threads\":", "\"n_params_m\":", "\"engine\":",
    };
    for (size_t i = 0; i < sizeof(required_fields) / sizeof(required_fields[0]); i++) {
        if (strstr(stdout_buf, required_fields[i]) == NULL) {
            fprintf(stderr,
                    "%s\nExpected JSON stdout to contain field %s\nActual stdout: %s\n",
                    label, required_fields[i], stdout_buf);
            assert(0);
        }
    }

    unlink(stdout_path);
    unlink(stderr_path);
    unlink(model_path);
    printf("%s: OK\n", label);
}

static void run_bench_malformed_metadata_case(const char *label,
                                              const bench_fixture_config_t *config,
                                              const char *expected_error) {
    char model_path[] = "/tmp/bitnet-bench-fixture-XXXXXX";
    char stderr_path[] = "/tmp/bitnet-bench-stderr-XXXXXX";
    char cmd[1024];
    char stderr_buf[4096];
    int fd;
    int status;

    create_tiny_bench_fixture(model_path, config);

    fd = mkstemp(stderr_path);
    assert(fd >= 0);
    close(fd);

    snprintf(cmd, sizeof(cmd), "./bitnet_bench -m %s >/dev/null 2>%s",
             model_path, stderr_path);
    status = system(cmd);
    assert(status != -1);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) != 0);

    read_file(stderr_path, stderr_buf, sizeof(stderr_buf));
    if (strstr(stderr_buf, expected_error) == NULL) {
        fprintf(stderr, "%s\nExpected stderr to contain: %s\nActual stderr: %s\n",
                label, expected_error, stderr_buf);
        assert(0);
    }

    unlink(stderr_path);
    unlink(model_path);
    printf("%s: OK\n", label);
}

static void run_cli_long_prompt_regression_case(void) {
    char model_path[] = "/tmp/bitnet-cli-fixture-XXXXXX";
    char *prompt = NULL;
    static const bench_fixture_config_t config = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        1,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };
    const char *argv[] = {
        "./bitnet_cli",
        "-m", model_path,
        "--ctx", "4096",
        "-n", "0",
        "-p", NULL,
        NULL,
    };

    create_tiny_bench_fixture(model_path, &config);

    prompt = (char *)malloc(5001);
    assert(prompt != NULL);
    memset(prompt, 'A', 5000);
    prompt[5000] = '\0';
    argv[8] = prompt;

    run_argv_case("cli rejects over-4096-token prompts instead of truncating",
                  argv,
                  "Error: prompt tokenization reached the CLI limit of 4096 tokens; prompt may be truncated");

    free(prompt);
    unlink(model_path);
}

int main(void) {
    static const bench_fixture_config_t malformed_tokens_config = {
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_STRING,
        1,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };
    static const bench_fixture_config_t malformed_merges_config = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_UINT32,
        1,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };
    static const bench_fixture_config_t malformed_bos_config = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        1,
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_UINT32,
    };
    static const bench_fixture_config_t malformed_eos_config = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        1,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_STRING,
    };

    printf("=== CLI Argument Validation Tests ===\n\n");

    run_case("cli rejects trailing junk for -n",
             "./bitnet_cli -n 12x",
             "Invalid value for -n: '12x' (must be a non-negative integer)");
    run_case("cli rejects negative -n",
             "./bitnet_cli -n -1",
             "Invalid value for -n: '-1' (must be a non-negative integer)");
    run_case("cli rejects invalid -t",
             "./bitnet_cli -t abc",
             "Invalid value for -t: 'abc' (must be a positive integer)");
    run_case("cli rejects non-finite temperature",
             "./bitnet_cli --temperature nan",
             "Invalid value for --temperature: 'nan' (must be a finite number)");
    run_case("cli rejects negative temperature",
             "./bitnet_cli --temperature -0.1",
             "Invalid value for --temperature: '-0.1' (must be a non-negative number)");
    run_case("cli rejects negative top-k",
             "./bitnet_cli --top-k -1",
             "Invalid value for --top-k: '-1' (must be a non-negative integer)");
    run_case("cli rejects invalid top-p",
             "./bitnet_cli --top-p 1.5",
             "Invalid value for --top-p: '1.5' (must be > 0 and <= 1)");
    run_case("cli rejects invalid ctx",
             "./bitnet_cli --ctx 0",
             "Invalid value for --ctx: '0' (must be a positive integer)");
    run_case("cli rejects overflowing seed",
             "./bitnet_cli --seed 18446744073709551616",
             "Invalid value for --seed: '18446744073709551616' (must be an unsigned 64-bit integer)");
    run_cli_long_prompt_regression_case();
    run_case("bench rejects invalid -t",
             "./bitnet_bench -t 0",
             "Invalid value for -t: '0' (must be a positive integer)");
    run_case("bench rejects unknown options",
             "./bitnet_bench --bogus",
             "Unknown option: --bogus");
    run_case("bench rejects missing -t value",
             "./bitnet_bench -t",
             "Option -t requires a value");
    run_bench_fixture_success_case("bench runs on small-vocab model without hardcoded seed");
    run_bench_fixture_error_case("bench exits cleanly on invalid derived seed token",
                                 "Prompt processing benchmark failed at token 0");
    run_bench_malformed_metadata_case("bench rejects non-string tokenizer token arrays",
                                      &malformed_tokens_config,
                                      "tokenizer.ggml.tokens must be an array of string, got array of uint32");
    run_bench_malformed_metadata_case("bench rejects non-string tokenizer merge arrays",
                                      &malformed_merges_config,
                                      "tokenizer.ggml.merges must be an array of string, got array of uint32");
    run_bench_malformed_metadata_case("bench rejects non-uint32 BOS token ids",
                                      &malformed_bos_config,
                                      "tokenizer.ggml.bos_token_id must have type uint32, got string");
    run_bench_malformed_metadata_case("bench rejects non-uint32 EOS token ids",
                                      &malformed_eos_config,
                                      "tokenizer.ggml.eos_token_id must have type uint32, got string");

    printf("\n=== All CLI argument validation tests passed ===\n");
    return 0;
}
