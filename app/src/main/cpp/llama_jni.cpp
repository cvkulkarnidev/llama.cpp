#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

#define LOG_TAG "LlamaCppGemma"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    llama_context_params ctx_params;
    std::mutex mutex;
    std::mutex log_mutex;
    bool backend_initialized = false;
    std::string loaded_backend = "CPU";
    std::string diagnostics;
    std::string accelerated_devices;
    int requested_gpu_layers = 0;
    int requested_context_size = 0;
    int requested_threads = 0;
    int accelerated_device_count = 0;
    int offloaded_layers = 0;
    bool gpu_offload_detected = false;
};

Engine g_engine;

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (!value) return "";
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars ? chars : "");
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring string_to_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

void throw_java(JNIEnv * env, const char * message) {
    auto clazz = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(clazz, message);
}

void append_diagnostic(const std::string & text) {
    std::lock_guard<std::mutex> lock(g_engine.log_mutex);
    g_engine.diagnostics += text;
    constexpr size_t max_diagnostics = 12000;
    if (g_engine.diagnostics.size() > max_diagnostics) {
        g_engine.diagnostics.erase(0, g_engine.diagnostics.size() - max_diagnostics);
    }
}

void clear_diagnostics() {
    std::lock_guard<std::mutex> lock(g_engine.log_mutex);
    g_engine.diagnostics.clear();
}

std::string diagnostics_snapshot() {
    std::lock_guard<std::mutex> lock(g_engine.log_mutex);
    return g_engine.diagnostics;
}

void llama_log_callback(enum ggml_log_level, const char * text, void *) {
    if (text) {
        append_diagnostic(text);
    }
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META: return "META";
    }
    return "UNKNOWN";
}

bool is_accelerated_device(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

std::string sanitize_field(std::string value) {
    std::replace(value.begin(), value.end(), ';', ',');
    std::replace(value.begin(), value.end(), '\n', ' ');
    return value;
}

int parse_offloaded_layers(const std::string & text) {
    int best = 0;
    size_t search_from = 0;
    const std::string marker = "offloaded ";

    while (true) {
        const size_t marker_pos = text.find(marker, search_from);
        if (marker_pos == std::string::npos) break;

        size_t digit_pos = marker_pos + marker.size();
        while (digit_pos < text.size() && !std::isdigit(static_cast<unsigned char>(text[digit_pos]))) {
            digit_pos++;
        }

        int parsed = 0;
        while (digit_pos < text.size() && std::isdigit(static_cast<unsigned char>(text[digit_pos]))) {
            parsed = parsed * 10 + (text[digit_pos] - '0');
            digit_pos++;
        }

        best = std::max(best, parsed);
        search_from = digit_pos;
    }

    return best;
}

void refresh_backend_devices() {
    g_engine.accelerated_device_count = 0;
    g_engine.accelerated_devices.clear();

    append_diagnostic("app: backend_registry_count=" + std::to_string(ggml_backend_reg_count()) + "\n");

    const size_t device_count = ggml_backend_dev_count();
    append_diagnostic("app: backend_device_count=" + std::to_string(device_count) + "\n");

    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        const char * name = ggml_backend_dev_name(device);
        const char * description = ggml_backend_dev_description(device);

        std::string label = name && name[0] ? name : "unknown";
        if (description && description[0]) {
            label += " (";
            label += description;
            label += ")";
        }

        append_diagnostic(
            "app: backend_device[" + std::to_string(i) + "]=" + label +
            " type=" + device_type_name(type) + "\n"
        );

        if (is_accelerated_device(type)) {
            if (!g_engine.accelerated_devices.empty()) {
                g_engine.accelerated_devices += ", ";
            }
            g_engine.accelerated_devices += label + " [" + device_type_name(type) + "]";
            g_engine.accelerated_device_count += 1;
        }
    }

    append_diagnostic("app: llama_supports_gpu_offload=" +
        std::string(llama_supports_gpu_offload() ? "true" : "false") + "\n");
}

void unload_locked() {
    if (g_engine.sampler) {
        llama_sampler_free(g_engine.sampler);
        g_engine.sampler = nullptr;
    }
    if (g_engine.ctx) {
        llama_free(g_engine.ctx);
        g_engine.ctx = nullptr;
    }
    if (g_engine.model) {
        llama_model_free(g_engine.model);
        g_engine.model = nullptr;
    }
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int token_count = -llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true
    );

    if (token_count <= 0) {
        throw std::runtime_error("Tokenization failed");
    }

    std::vector<llama_token> tokens(token_count);
    const int actual = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true
    );

    if (actual < 0) {
        throw std::runtime_error("Token buffer too small");
    }

    tokens.resize(actual);
    return tokens;
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    if (size < 0) {
        buffer.resize(static_cast<size_t>(-size));
        size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    }
    if (size <= 0) return "";
    return std::string(buffer.data(), static_cast<size_t>(size));
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_dev_chinmay_llamacppgemma_LlamaBridge_loadModel(
    JNIEnv * env,
    jobject,
    jstring model_path,
    jstring backend,
    jint gpu_layers,
    jint context_size,
    jint threads
) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    try {
        if (!g_engine.backend_initialized) {
            llama_log_set(llama_log_callback, nullptr);
            ggml_backend_load_all();
            llama_backend_init();
            g_engine.backend_initialized = true;
        }

        unload_locked();
        clear_diagnostics();

        const std::string path = jstring_to_string(env, model_path);
        const std::string backend_name = jstring_to_string(env, backend);
        g_engine.requested_gpu_layers = std::max(0, static_cast<int>(gpu_layers));
        g_engine.requested_context_size = std::max(512, static_cast<int>(context_size));
        g_engine.requested_threads = std::max(1, static_cast<int>(threads));
        append_diagnostic("app: native_build_type=" LLAMACPP_GEMMA_BUILD_TYPE "\n");
        append_diagnostic("app: backend_request=" + backend_name +
            " gpu_layers_requested=" + std::to_string(g_engine.requested_gpu_layers) +
            " context_size=" + std::to_string(g_engine.requested_context_size) +
            " threads=" + std::to_string(g_engine.requested_threads) + "\n");

        refresh_backend_devices();

        const bool wants_vulkan = backend_name == "vulkan" && g_engine.requested_gpu_layers > 0;
#ifndef LLAMACPP_GEMMA_VULKAN_ENABLED
        if (wants_vulkan) {
            throw std::runtime_error("Vulkan was requested, but this APK was not compiled with GGML_VULKAN.");
        }
#endif
        if (wants_vulkan && g_engine.accelerated_device_count <= 0) {
            throw std::runtime_error(
                "Vulkan was requested, but llama.cpp did not register a GPU/IGPU device. "
                "This APK would run CPU-only on this device."
            );
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = wants_vulkan ? g_engine.requested_gpu_layers : 0;

        g_engine.ctx_params = llama_context_default_params();
        g_engine.ctx_params.n_ctx = static_cast<uint32_t>(g_engine.requested_context_size);
        g_engine.ctx_params.n_threads = g_engine.requested_threads;
        g_engine.ctx_params.n_threads_batch = g_engine.requested_threads;
        g_engine.ctx_params.offload_kqv = wants_vulkan;
        g_engine.ctx_params.op_offload = wants_vulkan;
        g_engine.ctx_params.no_perf = false;

        g_engine.model = llama_model_load_from_file(path.c_str(), model_params);
        if (!g_engine.model) {
            throw std::runtime_error("Failed to load model. Check GGUF file and available RAM.");
        }

        g_engine.ctx = llama_init_from_model(g_engine.model, g_engine.ctx_params);
        if (!g_engine.ctx) {
            throw std::runtime_error("Failed to create llama context");
        }

        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = false;
        g_engine.sampler = llama_sampler_chain_init(sampler_params);
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        g_engine.offloaded_layers = parse_offloaded_layers(diagnostics_snapshot());
        g_engine.gpu_offload_detected = wants_vulkan && g_engine.offloaded_layers > 0;
        append_diagnostic("app: gpu_offload_detected=" +
            std::string(g_engine.gpu_offload_detected ? "true" : "false") +
            " offloaded_layers=" + std::to_string(g_engine.offloaded_layers) + "\n");

        if (backend_name == "cpu" || gpu_layers <= 0) {
            g_engine.loaded_backend = "CPU";
        } else if (backend_name == "vulkan") {
#ifdef LLAMACPP_GEMMA_VULKAN_ENABLED
            if (g_engine.gpu_offload_detected) {
                g_engine.loaded_backend = "Vulkan GPU active";
            } else {
                g_engine.loaded_backend = "Vulkan available - offload not confirmed";
                throw std::runtime_error(
                    "Vulkan was requested, but llama.cpp did not report any GPU layer offload. "
                    "This run was blocked to avoid silent CPU fallback."
                );
            }
#else
            g_engine.loaded_backend = "CPU - Vulkan not compiled into this APK";
#endif
        } else {
            g_engine.loaded_backend = "CPU";
        }

        LOGI("Loaded model %s with backend request %s and gpu_layers=%d", path.c_str(), backend_name.c_str(), gpu_layers);
        return string_to_jstring(env, g_engine.loaded_backend);
    } catch (const std::exception & e) {
        unload_locked();
        LOGE("%s", e.what());
        throw_java(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_chinmay_llamacppgemma_LlamaBridge_generate(
    JNIEnv * env,
    jobject,
    jstring prompt,
    jint max_tokens,
    jfloat temperature
) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    try {
        if (!g_engine.model || !g_engine.ctx) {
            throw std::runtime_error("Model is not loaded");
        }

        llama_free(g_engine.ctx);
        g_engine.ctx = llama_init_from_model(g_engine.model, g_engine.ctx_params);
        if (!g_engine.ctx) {
            throw std::runtime_error("Failed to reset llama context");
        }

        if (g_engine.sampler) {
            llama_sampler_free(g_engine.sampler);
        }
        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = false;
        g_engine.sampler = llama_sampler_chain_init(sampler_params);
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_temp(std::max(0.0f, static_cast<float>(temperature))));
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const llama_vocab * vocab = llama_model_get_vocab(g_engine.model);
        std::vector<llama_token> tokens = tokenize(vocab, jstring_to_string(env, prompt));

        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(g_engine.ctx, batch) != 0) {
            throw std::runtime_error("Prompt decode failed");
        }

        std::string output;
        const int limit = std::max(1, static_cast<int>(max_tokens));

        for (int i = 0; i < limit; ++i) {
            llama_token next = llama_sampler_sample(g_engine.sampler, g_engine.ctx, -1);
            if (llama_vocab_is_eog(vocab, next)) {
                break;
            }

            llama_sampler_accept(g_engine.sampler, next);
            output += token_to_piece(vocab, next);

            llama_batch next_batch = llama_batch_get_one(&next, 1);
            if (llama_decode(g_engine.ctx, next_batch) != 0) {
                break;
            }
        }

        return string_to_jstring(env, output);
    } catch (const std::exception & e) {
        LOGE("%s", e.what());
        throw_java(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_chinmay_llamacppgemma_LlamaBridge_benchmark(
    JNIEnv * env,
    jobject,
    jstring prompt,
    jint max_tokens,
    jfloat temperature
) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    try {
        if (!g_engine.model || !g_engine.ctx) {
            throw std::runtime_error("Model is not loaded");
        }

        llama_free(g_engine.ctx);
        g_engine.ctx = llama_init_from_model(g_engine.model, g_engine.ctx_params);
        if (!g_engine.ctx) {
            throw std::runtime_error("Failed to reset llama context for benchmark");
        }

        if (g_engine.sampler) {
            llama_sampler_free(g_engine.sampler);
        }
        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = false;
        g_engine.sampler = llama_sampler_chain_init(sampler_params);
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_temp(std::max(0.0f, static_cast<float>(temperature))));
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const llama_vocab * vocab = llama_model_get_vocab(g_engine.model);
        std::vector<llama_token> tokens = tokenize(vocab, jstring_to_string(env, prompt));

        std::vector<llama_token> warmup_tokens = tokenize(
            vocab,
            "<start_of_turn>user\nWarm up the local model briefly.\n<end_of_turn>\n<start_of_turn>model\n"
        );
        llama_batch warmup_batch = llama_batch_get_one(warmup_tokens.data(), static_cast<int32_t>(warmup_tokens.size()));
        if (llama_decode(g_engine.ctx, warmup_batch) != 0) {
            throw std::runtime_error("Benchmark warmup decode failed");
        }

        for (int i = 0; i < 4; ++i) {
            llama_token next = llama_sampler_sample(g_engine.sampler, g_engine.ctx, -1);
            if (llama_vocab_is_eog(vocab, next)) {
                break;
            }

            llama_sampler_accept(g_engine.sampler, next);
            llama_batch next_batch = llama_batch_get_one(&next, 1);
            if (llama_decode(g_engine.ctx, next_batch) != 0) {
                break;
            }
        }

        llama_free(g_engine.ctx);
        g_engine.ctx = llama_init_from_model(g_engine.model, g_engine.ctx_params);
        if (!g_engine.ctx) {
            throw std::runtime_error("Failed to reset llama context after benchmark warmup");
        }

        if (g_engine.sampler) {
            llama_sampler_free(g_engine.sampler);
        }
        sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = false;
        g_engine.sampler = llama_sampler_chain_init(sampler_params);
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_temp(std::max(0.0f, static_cast<float>(temperature))));
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        llama_perf_context_reset(g_engine.ctx);
        llama_perf_sampler_reset(g_engine.sampler);

        auto started = std::chrono::steady_clock::now();

        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(g_engine.ctx, batch) != 0) {
            throw std::runtime_error("Benchmark prompt decode failed");
        }
        auto prompt_finished = std::chrono::steady_clock::now();

        int generated = 0;
        const int limit = std::max(1, static_cast<int>(max_tokens));

        for (int i = 0; i < limit; ++i) {
            llama_token next = llama_sampler_sample(g_engine.sampler, g_engine.ctx, -1);
            if (llama_vocab_is_eog(vocab, next)) {
                break;
            }

            llama_sampler_accept(g_engine.sampler, next);
            generated += 1;

            llama_batch next_batch = llama_batch_get_one(&next, 1);
            if (llama_decode(g_engine.ctx, next_batch) != 0) {
                break;
            }
        }

        auto finished = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();
        const auto prompt_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prompt_finished - started).count();
        const auto generation_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - prompt_finished).count();
        const long long safe_elapsed_ms = std::max<long long>(1, static_cast<long long>(elapsed_ms));
        const long long safe_prompt_elapsed_ms = std::max<long long>(1, static_cast<long long>(prompt_elapsed_ms));
        const long long safe_generation_elapsed_ms = std::max<long long>(1, static_cast<long long>(generation_elapsed_ms));
        const auto perf = llama_perf_context(g_engine.ctx);
        append_diagnostic("app: benchmark_prompt_ms=" + std::to_string(safe_prompt_elapsed_ms) +
            " benchmark_decode_ms=" + std::to_string(safe_generation_elapsed_ms) +
            " llama_perf_prompt_ms=" + std::to_string(perf.t_p_eval_ms) +
            " llama_perf_eval_ms=" + std::to_string(perf.t_eval_ms) + "\n");
        const std::string result =
            "backend=" + g_engine.loaded_backend +
            ";build_type=" LLAMACPP_GEMMA_BUILD_TYPE +
            ";gpu_layers_requested=" + std::to_string(g_engine.requested_gpu_layers) +
            ";context_size=" + std::to_string(g_engine.requested_context_size) +
            ";threads=" + std::to_string(g_engine.requested_threads) +
            ";accelerated_device_count=" + std::to_string(g_engine.accelerated_device_count) +
            ";accelerated_devices=" + sanitize_field(g_engine.accelerated_devices) +
            ";gpu_offload_detected=" + std::string(g_engine.gpu_offload_detected ? "true" : "false") +
            ";offloaded_layers=" + std::to_string(g_engine.offloaded_layers) +
            ";prompt_tokens=" + std::to_string(tokens.size()) +
            ";generated_tokens=" + std::to_string(generated) +
            ";prompt_elapsed_ms=" + std::to_string(safe_prompt_elapsed_ms) +
            ";generation_elapsed_ms=" + std::to_string(safe_generation_elapsed_ms) +
            ";elapsed_ms=" + std::to_string(safe_elapsed_ms);

        LOGI("Benchmark %s", result.c_str());
        return string_to_jstring(env, result);
    } catch (const std::exception & e) {
        LOGE("%s", e.what());
        throw_java(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_chinmay_llamacppgemma_LlamaBridge_diagnostics(JNIEnv * env, jobject) {
    return string_to_jstring(env, diagnostics_snapshot());
}

extern "C" JNIEXPORT void JNICALL
Java_dev_chinmay_llamacppgemma_LlamaBridge_unload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    unload_locked();
}
