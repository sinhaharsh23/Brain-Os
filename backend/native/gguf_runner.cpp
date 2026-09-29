#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>

namespace {

struct TensorRecord {
    int layer;
    std::string kind;
    std::vector<float> values;
};

struct AttentionRecord {
    int layer;
    int head;
    std::vector<float> weights;
};

struct CaptureState {
    int step = 0;
    int current_positions = 0;
    int capture_limit = 256;
    int batch_start = 0;
    bool enabled = true;
    std::vector<TensorRecord> current;
    std::map<int, std::vector<TensorRecord>> recent_vectors;
    std::map<int, std::vector<AttentionRecord>> recent_attention;
};

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::string base_name(const char *raw) {
    std::string name = raw ? raw : "";
    for (const char *suffix : {" (view)", " (permuted)"}) {
        const size_t offset = name.find(suffix);
        if (offset != std::string::npos) name.resize(offset);
    }
    return name;
}

bool parse_layer(const std::string &name, int &layer) {
    const size_t dash = name.find_last_of('-');
    if (dash == std::string::npos || dash + 1 == name.size()) return false;
    try {
        layer = std::stoi(name.substr(dash + 1));
        return true;
    } catch (...) {
        return false;
    }
}

std::string kind_for(const std::string &name) {
    const std::string value = lower(name);
    if (value.rfind("qcur-", 0) == 0) return "q";
    if (value.rfind("kcur-", 0) == 0) return "k";
    if (value.rfind("vcur-", 0) == 0) return "v";
    if (value.rfind("kq_soft_max-", 0) == 0) return "attention";
    if (value.rfind("kq-", 0) == 0) return "scores";
    if (value.rfind("kqv-", 0) == 0) return "weighted_v";
    if (value.rfind("attn_out-", 0) == 0) return "attn_out";
    if (value.rfind("ffn_swiglu-", 0) == 0) return "ffn_activation";
    if (value.rfind("norm-", 0) == 0) return "layer_output";
    return "";
}

void emit_csv(const std::vector<float> &values) {
    std::cout << std::setprecision(7);
    for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
}

void emit_step(CaptureState &state) {
    auto &recent = state.recent_vectors[state.step];
    for (const auto &record : state.current) {
        const auto &values = record.values;
        if (values.empty()) continue;
        double sum = 0.0, squares = 0.0, variance = 0.0;
        float minimum = values.front(), maximum = values.front();
        for (float value : values) {
            sum += value;
            squares += static_cast<double>(value) * value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        const double mean = sum / values.size();
        for (float value : values) variance += (value - mean) * (value - mean);
        variance /= values.size();
        std::cout << "STAT\t" << state.step << '\t' << record.layer << '\t' << record.kind
                  << '\t' << values.size() << '\t' << minimum << '\t' << maximum << '\t'
                  << mean << '\t' << std::sqrt(variance) << '\t' << std::sqrt(squares) << '\t';
        const size_t sample_count = std::min<size_t>(8, values.size());
        for (size_t i = 0; i < sample_count; ++i) std::cout << (i ? "," : "") << values[i];
        if (record.kind == "ffn_activation") {
            std::vector<size_t> indices(values.size());
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
            const size_t top_count = std::min<size_t>(8, indices.size());
            std::partial_sort(indices.begin(), indices.begin() + top_count, indices.end(), [&](size_t a, size_t b) {
                return std::abs(values[a]) > std::abs(values[b]);
            });
            std::cout << '\t';
            for (size_t i = 0; i < top_count; ++i) {
                const size_t index = indices[i];
                std::cout << (i ? "," : "") << index << ':' << values[index];
            }
        }
        std::cout << '\n';
        if (record.kind == "q" || record.kind == "k" || record.kind == "v" ||
            record.kind == "attn_out" || record.kind == "layer_output") recent.push_back(record);
    }
    state.current.clear();

    for (const auto &attention : state.recent_attention[state.step]) {
        std::vector<size_t> indices(attention.weights.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        const size_t top_count = std::min<size_t>(8, indices.size());
        std::partial_sort(indices.begin(), indices.begin() + top_count, indices.end(), [&](size_t a, size_t b) {
            return attention.weights[a] > attention.weights[b];
        });
        // Publish the actual row before STEP_DONE announces its availability.
        std::cout << "ATTN_FULL\t" << state.step << '\t' << attention.layer << '\t' << attention.head << '\t';
        emit_csv(attention.weights);
        std::cout << '\n';
        std::cout << "ATTN\t" << state.step << '\t' << attention.layer << '\t' << attention.head << '\t';
        for (size_t i = 0; i < top_count; ++i) {
            const size_t index = indices[i];
            std::cout << (i ? "," : "") << index << ':' << attention.weights[index];
        }
        std::cout << '\n';
    }
    if (state.step > 8) {
        state.recent_vectors.erase(state.step - 8);
        state.recent_attention.erase(state.step - 8);
    }
    std::cout << "STEP_DONE\t" << state.step << '\n' << std::flush;
}

void emit_details(const CaptureState &state) {
    for (const auto &[step, records] : state.recent_vectors) {
        for (const auto &record : records) {
            std::cout << "VEC\t" << step << '\t' << record.layer << '\t' << record.kind << '\t';
            emit_csv(record.values);
            std::cout << '\n';
        }
    }
    std::cout << std::flush;
}

bool capture_callback(ggml_tensor *tensor, bool ask, void *user_data) {
    auto *state = static_cast<CaptureState *>(user_data);
    const char *raw_name = ggml_get_name(tensor);
    const std::string name = base_name(raw_name);
    const std::string kind = kind_for(name);
    const bool view = raw_name && (std::strstr(raw_name, "(view)") || std::strstr(raw_name, "(permuted)"));
    // embd is the real token lookup output, before transformer layers.
    // Capture every prefill batch, including batches without layer capture.
    if ((name == "embd" || name == "inp_embd") && !view && !ask && tensor->type == GGML_TYPE_F32) {
        std::vector<float> values(static_cast<size_t>(tensor->ne[0]));
        for (int64_t row = 0; row < tensor->ne[1]; ++row) {
            ggml_backend_tensor_get(tensor, values.data(), row * tensor->nb[1], values.size() * sizeof(float));
            std::cout << "EMBED\t" << state->batch_start + row << '\t';
            emit_csv(values);
            std::cout << '\n';
        }
        return true;
    }
    if (kind.empty() || !state->enabled || view || ask || tensor->type != GGML_TYPE_F32 || !tensor->data) return true;
    int layer = -1;
    if (!parse_layer(name, layer)) return true;
    const int64_t dim0 = tensor->ne[0], dim1 = tensor->ne[1], dim2 = tensor->ne[2];
    if (dim1 <= 0) return true;
    const float *data = static_cast<const float *>(tensor->data);
    const int64_t row = dim1 - 1;

    if (kind == "scores") return true;
    if (kind == "attention") {
        const int64_t keys = std::min<int64_t>(std::min<int64_t>(dim0, state->capture_limit), state->current_positions);
        for (int64_t head = 0; head < dim2; ++head) {
            const float *start = data + head * dim0 * dim1 + row * dim0;
            state->recent_attention[state->step].push_back({layer, static_cast<int>(head), std::vector<float>(start, start + keys)});
        }
        return true;
    }
    if (kind == "weighted_v") return true;
    const float *start = data + row * dim0;
    state->current.push_back({layer, kind, std::vector<float>(start, start + dim0)});
    return true;
}

int32_t apply_template(const llama_model *model, const std::string &prompt, std::vector<char> &formatted) {
    const char *tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        formatted.assign(prompt.begin(), prompt.end());
        return static_cast<int32_t>(formatted.size());
    }
    const llama_chat_message message{"user", prompt.c_str()};
    int32_t capacity = static_cast<int32_t>(prompt.size() * 4 + 1024);
    formatted.resize(static_cast<size_t>(capacity));
    int32_t size = llama_chat_apply_template(tmpl, &message, 1, true, formatted.data(), capacity);
    if (size < 0 || size > capacity) {
        capacity = size + 1;
        formatted.resize(static_cast<size_t>(capacity));
        size = llama_chat_apply_template(tmpl, &message, 1, true, formatted.data(), capacity);
    }
    if (size < 0) {
        formatted.assign(prompt.begin(), prompt.end());
        return static_cast<int32_t>(formatted.size());
    }
    formatted.resize(static_cast<size_t>(size));
    return size;
}

std::string token_piece(const llama_vocab *vocab, llama_token token) {
    char buffer[256] = {};
    const int32_t size = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, true);
    return size < 0 ? "" : std::string(buffer, static_cast<size_t>(size));
}

std::string hex_piece(const std::string &value) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char c : value) { result.push_back(digits[c >> 4]); result.push_back(digits[c & 15]); }
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s MODEL.gguf MAX_NEW_TOKENS PROMPT\n", argv[0]); return 2; }
    const int max_new = std::max(1, std::min(1200, std::atoi(argv[2])));
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) { llama_backend_free(); std::fprintf(stderr, "failed to load GGUF model\n"); return 3; }

    std::vector<char> formatted;
    const int32_t formatted_size = apply_template(model, argv[3], formatted);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(static_cast<size_t>(std::max(64, formatted_size * 2 + 32)));
    int32_t token_count = llama_tokenize(vocab, formatted.data(), formatted_size, tokens.data(), static_cast<int32_t>(tokens.size()), false, true);
    if (token_count < 0) { tokens.resize(static_cast<size_t>(-token_count)); token_count = llama_tokenize(vocab, formatted.data(), formatted_size, tokens.data(), static_cast<int32_t>(tokens.size()), false, true); }
    if (token_count <= 0) { llama_model_free(model); llama_backend_free(); std::fprintf(stderr, "tokenization failed\n"); return 4; }
    tokens.resize(static_cast<size_t>(token_count));

    CaptureState capture;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(std::min(2048, std::max(512, token_count + max_new + 8)));
    context_params.n_batch = static_cast<uint32_t>(std::min(512, token_count));
    context_params.n_ubatch = context_params.n_batch;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.cb_eval = capture_callback;
    context_params.cb_eval_user_data = &capture;
    llama_context *context = llama_init_from_model(model, context_params);
    if (!context) { llama_model_free(model); llama_backend_free(); std::fprintf(stderr, "failed to create context\n"); return 5; }

    const int32_t layers = llama_model_n_layer(model), hidden = llama_model_n_embd(model);
    const int32_t q_heads = llama_model_n_head(model), kv_heads = llama_model_n_head_kv(model);
    const int32_t head_dim = hidden / std::max(1, q_heads);
    std::cout << "META\t" << layers << '\t' << hidden << '\t' << q_heads << '\t' << kv_heads << '\t' << head_dim << '\t' << llama_vocab_n_tokens(vocab) << '\t' << llama_model_n_params(model) << '\t' << llama_model_n_ctx_train(model) << '\t' << llama_model_size(model) << '\n';
    std::cout << "PROMPT\t" << token_count << '\n';
    for (int32_t i = 0; i < token_count; ++i) std::cout << "PTOKEN\t" << i << '\t' << tokens[i] << '\t' << hex_piece(token_piece(vocab, tokens[i])) << '\n';
    std::cout << "PREFILL\n" << std::flush;

    llama_batch batch = llama_batch_init(static_cast<int32_t>(context_params.n_batch), 0, 1);
    int decode_result = 0;
    for (int32_t offset = 0; offset < token_count; offset += static_cast<int32_t>(context_params.n_batch)) {
        const int32_t count = std::min<int32_t>(static_cast<int32_t>(context_params.n_batch), token_count - offset);
        for (int32_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[offset + i]; batch.pos[i] = offset + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = offset + i == token_count - 1;
        }
        capture.batch_start = offset;
        batch.n_tokens = count; capture.step = 0; capture.current_positions = offset + count; capture.enabled = offset + count == token_count;
        std::cout << "STEP_START\t0\n" << std::flush;
        decode_result = llama_decode(context, batch);
        if (decode_result != 0) break;
    }
    capture.enabled = true;
    if (decode_result != 0) { llama_batch_free(batch); llama_free(context); llama_model_free(model); llama_backend_free(); std::fprintf(stderr, "prompt decode failed: %d\n", decode_result); return 6; }
    emit_step(capture);

    llama_sampler *sampler = llama_sampler_init_greedy();
    int32_t generated = 0;
    for (; generated < max_new; ++generated) {
        float *logits = llama_get_logits_ith(context, -1);
        if (!logits) break;
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        llama_token next = 0; float max_logit = logits[0];
        for (int32_t i = 1; i < vocab_size; ++i) if (logits[i] > max_logit) { max_logit = logits[i]; next = i; }
        double denom = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) denom += std::exp(static_cast<double>(logits[i] - max_logit));
        std::vector<int32_t> order(static_cast<size_t>(vocab_size));
        for (int32_t i = 0; i < vocab_size; ++i) order[static_cast<size_t>(i)] = i;
        const size_t top = std::min<size_t>(8, order.size());
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        for (size_t rank = 0; rank < top; ++rank) {
            const int32_t id = order[rank];
            std::cout << "CAND\t" << generated << '\t' << rank + 1 << '\t' << id << '\t' << logits[id] << '\t' << std::exp(static_cast<double>(logits[id] - max_logit)) / denom << '\t' << hex_piece(token_piece(vocab, id)) << '\n';
        }
        if (llama_vocab_is_eog(vocab, next)) break;
        const double probability = std::exp(static_cast<double>(logits[next] - max_logit)) / denom;
        std::cout << "TOKEN\t" << generated << '\t' << next << '\t' << hex_piece(token_piece(vocab, next)) << '\t' << probability << '\n' << std::flush;
        if (generated + 1 >= max_new) { ++generated; break; }
        llama_sampler_accept(sampler, next);
        capture.batch_start = token_count + generated;
        capture.step = generated + 1; capture.current_positions = token_count + generated + 1;
        batch.n_tokens = 1; batch.token[0] = next; batch.pos[0] = token_count + generated; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        std::cout << "STEP_START\t" << capture.step << '\n' << std::flush;
        decode_result = llama_decode(context, batch);
        if (decode_result != 0) break;
        emit_step(capture);
    }
    emit_details(capture);
    std::cout << "DONE\t" << generated << '\t' << token_count << '\t' << decode_result << '\n' << std::flush;
    llama_sampler_free(sampler); llama_batch_free(batch); llama_free(context); llama_model_free(model); llama_backend_free();
    return 0;
}
