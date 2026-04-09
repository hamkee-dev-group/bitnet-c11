#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef MODEL_PATH
#define MODEL_PATH "models/ggml-model-i2_s.gguf"
#endif

int main(void) {
    const char *model_path = getenv("BITNET_MODEL");
    if (!model_path) model_path = MODEL_PATH;

    printf("=== Tokenizer Tests ===\n\n");

    bn_gguf_t *g = bn_gguf_open(model_path);
    assert(g != NULL);

    bn_tokenizer_t *tok = bn_tokenizer_create(g);
    assert(tok != NULL);

    printf("Test 1: Encode 'Hello'...\n");
    int tokens[256];
    int n = bn_tokenize(tok, "Hello", tokens, 256);
    printf("  Tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
    printf("\n");
    assert(n >= 2);
    assert(tokens[0] == bn_token_bos(tok));
    printf("  OK\n");

    printf("Test 2: Decode...\n");
    char *text = bn_detokenize(tok, tokens + 1, n - 1);
    printf("  Decoded: '%s'\n", text);
    assert(strcmp(text, "Hello") == 0);
    free(text);
    printf("  OK\n");

    printf("Test 3: 'The capital of France'...\n");
    n = bn_tokenize(tok, "The capital of France", tokens, 256);
    printf("  Tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
    printf("\n");
    text = bn_detokenize(tok, tokens + 1, n - 1);
    printf("  Decoded: '%s'\n", text);
    assert(strcmp(text, "The capital of France") == 0);
    free(text);
    printf("  OK\n");

    printf("Test 4: Special tokens...\n");
    printf("  BOS: %d, EOS: %d\n", bn_token_bos(tok), bn_token_eos(tok));
    assert(bn_token_bos(tok) == 128000);
    assert(bn_token_eos(tok) == 128001);
    printf("  OK\n");

    printf("Test 5: Token text...\n");
    const char *hello_text = bn_token_text(tok, tokens[1]);
    printf("  Token %d = '%s'\n", tokens[1], hello_text);
    assert(strlen(hello_text) > 0);
    printf("  OK\n");

    printf("Test 6: Numbers '12345'...\n");
    n = bn_tokenize(tok, "12345", tokens, 256);
    printf("  Tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
    printf("\n");
    text = bn_detokenize(tok, tokens + 1, n - 1);
    printf("  Decoded: '%s'\n", text);
    assert(strcmp(text, "12345") == 0);
    free(text);
    printf("  OK\n");

    bn_tokenizer_free(tok);
    bn_gguf_close(g);

    printf("\n=== All tokenizer tests passed ===\n");
    return 0;
}
