#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

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
    bool backend_initialized = false;
    std::string loaded_backend = "CPU";
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
            ggml_backend_load_all();
            llama_backend_init();
            g_engine.backend_initialized = true;
        }

        unload_locked();

        const std::string path = jstring_to_string(env, model_path);
        const std::string backend_name = jstring_to_string(env, backend);

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = std::max(0, static_cast<int>(gpu_layers));

        g_engine.ctx_params = llama_context_default_params();
        g_engine.ctx_params.n_ctx = static_cast<uint32_t>(std::max(512, static_cast<int>(context_size)));
        g_engine.ctx_params.n_threads = std::max(1, static_cast<int>(threads));
        g_engine.ctx_params.n_threads_batch = std::max(1, static_cast<int>(threads));

        g_engine.model = llama_model_load_from_file(path.c_str(), model_params);
        if (!g_engine.model) {
            throw std::runtime_error("Failed to load model. Check GGUF file and available RAM.");
        }

        g_engine.ctx = llama_init_from_model(g_engine.model, g_engine.ctx_params);
        if (!g_engine.ctx) {
            throw std::runtime_error("Failed to create llama context");
        }

        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        g_engine.sampler = llama_sampler_chain_init(sampler_params);
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        if (backend_name == "cpu" || gpu_layers <= 0) {
            g_engine.loaded_backend = "CPU";
        } else if (backend_name == "qnn") {
            g_engine.loaded_backend = "QNN/NPU if compiled, otherwise llama.cpp fallback";
        } else if (backend_name == "opencl") {
            g_engine.loaded_backend = "OpenCL GPU if compiled, otherwise llama.cpp fallback";
        } else {
            g_engine.loaded_backend = "Vulkan GPU if compiled, otherwise llama.cpp fallback";
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

extern "C" JNIEXPORT void JNICALL
Java_dev_chinmay_llamacppgemma_LlamaBridge_unload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    unload_locked();
}
