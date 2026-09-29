#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>

namespace {

struct CaptureState {
    int captured = 0;
};

bool is_capture_tensor(const char *name) {
    if (name == nullptr) return false;
    std::string value(name);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
        return value.find("attn") != std::string::npos ||
            value.find("kq") != std::string::npos ||
            value.find("qcur") != std::string::npos ||
            value.find("kcur") != std::string::npos ||
            value.find("vcur") != std::string::npos ||
            value.find("soft") != std::string::npos ||
            value.find("ffn") != std::string::npos ||
            value.find("norm") != std::string::npos ||
            value.find("result") != std::string::npos;
}

bool capture_callback(ggml_tensor *tensor, bool ask, void *user_data) {
    auto *state = static_cast<CaptureState *>(user_data);
    const char *name = ggml_get_name(tensor);
    if (!is_capture_tensor(name)) return true;
    if (ask) return true;

    std::printf("TENSOR name=%s type=%s dims=%d shape=[%lld,%lld,%lld,%lld] bytes=%zu",
                name,
                ggml_type_name(tensor->type),
                ggml_n_dims(tensor),
                static_cast<long long>(tensor->ne[0]),
                static_cast<long long>(tensor->ne[1]),
                static_cast<long long>(tensor->ne[2]),
                static_cast<long long>(tensor->ne[3]),
                ggml_nbytes(tensor));

    const int64_t elements = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32 && elements > 0) {
        float sample[8] = {};
        const size_t count = static_cast<size_t>(std::min<int64_t>(elements, 8));
        if (tensor->buffer != nullptr) {
            ggml_backend_tensor_get(tensor, sample, 0, count * sizeof(float));
        } else if (tensor->data != nullptr) {
            std::memcpy(sample, tensor->data, count * sizeof(float));
        }
        std::printf(" sample=[");
        for (size_t i = 0; i < count; ++i) {
            std::printf("%s%.6f", i == 0 ? "" : ",", sample[i]);
        }
        std::printf("]");
    }
    std::printf("\n");
    ++state->captured;
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf PROMPT\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load GGUF model\n");
        llama_backend_free();
        return 3;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t token_capacity = static_cast<int32_t>(std::strlen(argv[2]) * 2 + 32);
    auto *tokens = new llama_token[token_capacity];
    int32_t token_count = llama_tokenize(vocab, argv[2], static_cast<int32_t>(std::strlen(argv[2])), tokens, token_capacity, true, true);
    if (token_count < 0) {
        delete[] tokens;
        llama_model_free(model);
        llama_backend_free();
        std::fprintf(stderr, "tokenization failed: %d\n", token_count);
        return 4;
    }

    CaptureState capture_state;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(std::max<int32_t>(512, token_count + 16));
    context_params.n_batch = static_cast<uint32_t>(token_count);
    context_params.n_ubatch = static_cast<uint32_t>(token_count);
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.cb_eval = capture_callback;
    context_params.cb_eval_user_data = &capture_state;
    llama_context *context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        delete[] tokens;
        llama_model_free(model);
        llama_backend_free();
        std::fprintf(stderr, "failed to create llama context\n");
        return 5;
    }

    llama_batch batch = llama_batch_init(token_count, 0, 1);
    batch.n_tokens = token_count;
    for (int32_t i = 0; i < token_count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == token_count - 1;
    }

    const int decode_result = llama_decode(context, batch);
    std::printf("SUMMARY tokens=%d captured_tensors=%d decode_result=%d\n",
                token_count, capture_state.captured, decode_result);

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    delete[] tokens;
    llama_backend_free();
    return decode_result == 0 && capture_state.captured > 0 ? 0 : 6;
}