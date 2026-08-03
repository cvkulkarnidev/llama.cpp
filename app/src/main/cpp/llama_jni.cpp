#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
    std::mutex log_mutex;
    bool backend_initialized = false;
    std::string loaded_backend = "CPU";
    std::string diagnostics;
    int requested_gpu_layers = 0;
    int requested_context_size = 0;
    int requested_threads = 0;
    std::vector<llama_token> cached_tokens;
};

Engine g_engine;

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (!value) return "";
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) {
        throw std::runtime_error("Unable to read Java string");
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

std::vector<std::string> jobject_array_to_strings(JNIEnv * env, jobjectArray values) {
    if (!values) {
        throw std::runtime_error("Missing chat messages");
    }

    const jsize count = env->GetArrayLength(values);
    std::vector<std::string> result;
    result.reserve(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        auto value = static_cast<jstring>(env->GetObjectArrayElement(values, i));
        if (!value) {
            throw std::runtime_error("Chat messages cannot contain null values");
        }
        result.push_back(jstring_to_string(env, value));
        env->DeleteLocalRef(value);
    }
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

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "integrated GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        case GGML_BACKEND_DEVICE_TYPE_META: return "meta";
    }
    return "unknown";
}

std::string device_description(ggml_backend_dev_t device) {
    const char * description = ggml_backend_dev_description(device);
    if (description && description[0] != '\0') {
        return description;
    }
    const char * name = ggml_backend_dev_name(device);
    return name ? name : "unknown device";
}

std::string backend_name(ggml_backend_dev_t device) {
    auto registry = ggml_backend_dev_backend_reg(device);
    const char * name = registry ? ggml_backend_reg_name(registry) : nullptr;
    return name ? name : "unknown";
}

void append_device_inventory() {
    append_diagnostic("app: available_devices=" + std::to_string(ggml_backend_dev_count()) + "\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        append_diagnostic(
            "app: device=" + std::string(ggml_backend_dev_name(device)) +
            " backend=" + backend_name(device) +
            " type=" + device_type_name(ggml_backend_dev_type(device)) +
            " description=" + device_description(device) + "\n"
        );
    }
}

ggml_backend_dev_t find_vulkan_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        if (lowercase(backend_name(device)) == "vulkan") {
            return device;
        }
    }
    return nullptr;
}

void unload_locked() {
    g_engine.cached_tokens.clear();
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
    char stack_buffer[256];
    int size = llama_token_to_piece(vocab, token, stack_buffer, sizeof(stack_buffer), 0, true);
    if (size < 0) {
        std::vector<char> buffer(static_cast<size_t>(-size));
        size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
        if (size <= 0) return "";
        return std::string(buffer.data(), static_cast<size_t>(size));
    }
    if (size <= 0) return "";
    return std::string(stack_buffer, static_cast<size_t>(size));
}

std::string format_gemma4_chat_prompt(
    const std::vector<std::string> & roles,
    const std::vector<std::string> & contents
) {
    std::string formatted;
    for (size_t i = 0; i < roles.size(); ++i) {
        const std::string role = roles[i] == "assistant" ? "model" : roles[i];
        formatted += "<|turn>" + role + "\n" + contents[i] + "<turn|>\n";
    }
    formatted += "<|turn>model\n";
    return formatted;
}

std::string format_chat_prompt(
    const llama_model * model,
    const std::vector<std::string> & roles,
    const std::vector<std::string> & contents
) {
    if (roles.empty() || roles.size() != contents.size()) {
        throw std::runtime_error("Chat roles and messages must have the same non-zero length");
    }

    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        if (roles[i] != "system" && roles[i] != "user" && roles[i] != "assistant") {
            throw std::runtime_error("Unsupported chat role: " + roles[i]);
        }
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }

    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) {
        throw std::runtime_error("This GGUF does not contain a chat template");
    }

    int32_t required = llama_chat_apply_template(
        chat_template,
        messages.data(),
        messages.size(),
        true,
        nullptr,
        0
    );
    if (required < 0) {
        const std::string template_text(chat_template);
        if (
            template_text.find("<|turn>") != std::string::npos &&
            template_text.find("<turn|>") != std::string::npos
        ) {
            append_diagnostic("app: chat_template_fallback=gemma4_text\n");
            return format_gemma4_chat_prompt(roles, contents);
        }
        throw std::runtime_error("The GGUF chat template is not supported by this llama.cpp build");
    }

    std::vector<char> formatted(static_cast<size_t>(required));
    const int32_t written = llama_chat_apply_template(
        chat_template,
        messages.data(),
        messages.size(),
        true,
        formatted.data(),
        static_cast<int32_t>(formatted.size())
    );
    if (written < 0 || written > required) {
        throw std::runtime_error("Failed to apply the GGUF chat template");
    }
    return std::string(formatted.data(), static_cast<size_t>(written));
}

void reset_sampler(float temperature) {
    if (g_engine.sampler) {
        llama_sampler_free(g_engine.sampler);
    }
    g_engine.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
}

struct InferenceResult {
    std::string output;
    int prompt_tokens = 0;
    int cached_prompt_tokens = 0;
    int generated_tokens = 0;
    long long prompt_eval_ms = 0;
    long long generation_ms = 0;
    long long time_to_first_token_ms = 0;
    long long total_ms = 0;
};

using StreamCallback = std::function<void(const InferenceResult &)>;

InferenceResult run_inference(
    const std::string & prompt,
    int max_tokens,
    float temperature,
    bool capture_output,
    const StreamCallback & on_token = {},
    std::chrono::steady_clock::time_point request_started = std::chrono::steady_clock::now()
) {
    reset_sampler(std::max(0.0f, temperature));

    const llama_vocab * vocab = llama_model_get_vocab(g_engine.model);
    std::vector<llama_token> tokens = tokenize(vocab, prompt);
    const int context_size = static_cast<int>(llama_n_ctx(g_engine.ctx));
    const int available_tokens = context_size - static_cast<int>(tokens.size());
    if (available_tokens <= 0) {
        throw std::runtime_error(
            "Prompt uses " + std::to_string(tokens.size()) +
            " tokens but the context size is " + std::to_string(context_size)
        );
    }

    const int requested_limit = std::max(1, max_tokens);
    const int limit = std::min(requested_limit, available_tokens);
    if (limit < requested_limit) {
        append_diagnostic(
            "app: max_tokens_capped=" + std::to_string(limit) +
            " requested=" + std::to_string(requested_limit) + "\n"
        );
    }

    InferenceResult result;
    result.prompt_tokens = static_cast<int>(tokens.size());

    size_t common_prefix = 0;
    const size_t comparable = std::min(tokens.size(), g_engine.cached_tokens.size());
    while (common_prefix < comparable && tokens[common_prefix] == g_engine.cached_tokens[common_prefix]) {
        ++common_prefix;
    }
    // At least one token must be evaluated so the context exposes logits for sampling.
    if (common_prefix == tokens.size() && common_prefix > 0) {
        --common_prefix;
    }

    llama_memory_t memory = llama_get_memory(g_engine.ctx);
    if (common_prefix == 0) {
        llama_memory_clear(memory, false);
        g_engine.cached_tokens.clear();
    } else {
        if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(common_prefix), -1)) {
            llama_memory_clear(memory, false);
            common_prefix = 0;
        }
        g_engine.cached_tokens.resize(common_prefix);
    }
    result.cached_prompt_tokens = static_cast<int>(common_prefix);

    const auto prompt_started = std::chrono::steady_clock::now();
    size_t processed = common_prefix;
    const size_t prompt_batch_size = static_cast<size_t>(llama_n_batch(g_engine.ctx));
    while (processed < tokens.size()) {
        const size_t chunk_size = std::min(prompt_batch_size, tokens.size() - processed);
        llama_batch batch = llama_batch_get_one(
            tokens.data() + processed,
            static_cast<int32_t>(chunk_size)
        );
        if (llama_decode(g_engine.ctx, batch) != 0) {
            throw std::runtime_error("Prompt decode failed");
        }
        processed += chunk_size;
    }
    g_engine.cached_tokens.insert(
        g_engine.cached_tokens.end(),
        tokens.begin() + static_cast<std::ptrdiff_t>(common_prefix),
        tokens.end()
    );
    const auto prompt_finished = std::chrono::steady_clock::now();
    result.prompt_eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        prompt_finished - prompt_started
    ).count();

    const auto generation_started = prompt_finished;
    auto last_streamed_at = generation_started;
    int last_streamed_tokens = 0;
    constexpr auto stream_interval = std::chrono::milliseconds(32);

    const auto stream = [&](bool force) {
        if (!on_token || result.generated_tokens == last_streamed_tokens) return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_streamed_at < stream_interval) return;
        result.generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - generation_started
        ).count();
        result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - request_started
        ).count();
        on_token(result);
        last_streamed_at = now;
        last_streamed_tokens = result.generated_tokens;
    };

    for (int i = 0; i < limit; ++i) {
        llama_token next = llama_sampler_sample(g_engine.sampler, g_engine.ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            stream(true);
            break;
        }

        if (capture_output || on_token) {
            result.output += token_to_piece(vocab, next);
        }
        result.generated_tokens += 1;
        if (result.generated_tokens == 1) {
            result.time_to_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - request_started
            ).count();
            stream(true);
        }

        // The final predicted token does not need a decode because no subsequent sample uses it.
        if (i + 1 == limit) {
            stream(true);
            break;
        }

        llama_batch next_batch = llama_batch_get_one(&next, 1);
        if (llama_decode(g_engine.ctx, next_batch) != 0) {
            throw std::runtime_error("Token decode failed");
        }
        g_engine.cached_tokens.push_back(next);
        stream(false);
    }
    const auto generation_finished = std::chrono::steady_clock::now();

    result.generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        generation_finished - generation_started
    ).count();
    result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        generation_finished - request_started
    ).count();
    stream(true);
    return result;
}

long long warmup_context() {
    const auto started = std::chrono::steady_clock::now();
    const llama_vocab * vocab = llama_model_get_vocab(g_engine.model);
    std::vector<llama_token> warmup_tokens;
    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    if (bos != LLAMA_TOKEN_NULL) warmup_tokens.push_back(bos);
    if (eos != LLAMA_TOKEN_NULL) warmup_tokens.push_back(eos);
    if (warmup_tokens.empty()) warmup_tokens.push_back(0);

    llama_memory_t memory = llama_get_memory(g_engine.ctx);
    if (llama_model_has_decoder(g_engine.model) && !llama_model_has_encoder(g_engine.model)) {
        if (llama_decode(
                g_engine.ctx,
                llama_batch_get_one(warmup_tokens.data(), static_cast<int32_t>(warmup_tokens.size()))
            ) != 0) {
            throw std::runtime_error("Model warmup prompt decode failed");
        }
        llama_memory_clear(memory, false);
        if (llama_decode(g_engine.ctx, llama_batch_get_one(warmup_tokens.data(), 1)) != 0) {
            throw std::runtime_error("Model warmup token decode failed");
        }
    }
    llama_synchronize(g_engine.ctx);
    llama_memory_clear(memory, false);
    g_engine.cached_tokens.clear();

    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started
    ).count();
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
        const auto load_started = std::chrono::steady_clock::now();
        unload_locked();
        clear_diagnostics();

        if (!g_engine.backend_initialized) {
            llama_log_set(llama_log_callback, nullptr);
            ggml_backend_load_all();
            llama_backend_init();
            g_engine.backend_initialized = true;
        }
        append_device_inventory();

        const std::string path = jstring_to_string(env, model_path);
        const std::string backend_name = jstring_to_string(env, backend);
        if (path.empty()) {
            throw std::runtime_error("Model path is empty");
        }
        if (backend_name != "cpu" && backend_name != "vulkan") {
            throw std::runtime_error("Unsupported backend: " + backend_name);
        }

        g_engine.requested_gpu_layers = backend_name == "vulkan"
            ? std::max(1, static_cast<int>(gpu_layers))
            : 0;
        g_engine.requested_context_size = std::max(512, static_cast<int>(context_size));
        g_engine.requested_threads = std::max(1, static_cast<int>(threads));
        append_diagnostic("app: native_build_type=" LLAMACPP_GEMMA_BUILD_TYPE "\n");
        append_diagnostic("app: llama_cpp_revision=" LLAMACPP_GEMMA_LLAMA_REVISION "\n");
        append_diagnostic("app: backend_request=" + backend_name +
            " gpu_layers_requested=" + std::to_string(g_engine.requested_gpu_layers) +
            " context_size=" + std::to_string(g_engine.requested_context_size) +
            " threads=" + std::to_string(g_engine.requested_threads) + "\n");

        std::vector<ggml_backend_dev_t> selected_devices;
        std::string selected_backend;
        if (backend_name == "cpu") {
            selected_devices.push_back(nullptr);
            selected_backend = "CPU";
        } else {
#ifdef LLAMACPP_GEMMA_VULKAN_ENABLED
            auto vulkan_device = find_vulkan_device();
            if (!vulkan_device) {
                throw std::runtime_error("Vulkan was compiled into the APK but no Vulkan device is available");
            }
            selected_devices.push_back(vulkan_device);
            selected_devices.push_back(nullptr);
            selected_backend = "Vulkan GPU: " + device_description(vulkan_device);
            append_diagnostic(
                "app: selected_device=" + std::string(ggml_backend_dev_name(vulkan_device)) +
                " description=" + device_description(vulkan_device) + "\n"
            );
#else
            throw std::runtime_error("Vulkan was not compiled into this APK");
#endif
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = g_engine.requested_gpu_layers;
        model_params.devices = selected_devices.data();

        g_engine.ctx_params = llama_context_default_params();
        g_engine.ctx_params.n_ctx = static_cast<uint32_t>(g_engine.requested_context_size);
        g_engine.ctx_params.n_batch = static_cast<uint32_t>(std::min(2048, g_engine.requested_context_size));
        g_engine.ctx_params.n_ubatch = static_cast<uint32_t>(std::min(512, g_engine.requested_context_size));
        g_engine.ctx_params.n_seq_max = 1;
        g_engine.ctx_params.n_outputs_max = 1;
        g_engine.ctx_params.n_threads = g_engine.requested_threads;
        const int available_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        g_engine.ctx_params.n_threads_batch = std::min(8, std::max(g_engine.requested_threads, available_threads));
        g_engine.ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        g_engine.ctx_params.offload_kqv = true;
        g_engine.ctx_params.op_offload = true;
        g_engine.ctx_params.no_perf = true;
        g_engine.ctx_params.swa_full = false;

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

        const auto initialized_at = std::chrono::steady_clock::now();
        const long long warmup_ms = warmup_context();
        const long long model_init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            initialized_at - load_started
        ).count();

        g_engine.loaded_backend = selected_backend;
        append_diagnostic("app: model_loaded_on=" + selected_backend + "\n");
        append_diagnostic(
            "app: optimizations=persistent_context,prefix_kv_cache,batched_streaming,last_decode_elision,"
            "flash_attention_auto,op_offload,kv_offload,armv8.7a,lto\n"
        );
        append_diagnostic(
            "app: actual_context_size=" + std::to_string(llama_n_ctx(g_engine.ctx)) +
            " batch=" + std::to_string(llama_n_batch(g_engine.ctx)) +
            " ubatch=" + std::to_string(llama_n_ubatch(g_engine.ctx)) +
            " threads_batch=" + std::to_string(g_engine.ctx_params.n_threads_batch) + "\n"
        );
        append_diagnostic(
            "app: model_init_ms=" + std::to_string(model_init_ms) +
            " warmup_ms=" + std::to_string(warmup_ms) + "\n"
        );

        LOGI(
            "Loaded model %s on %s with gpu_layers=%d",
            path.c_str(),
            selected_backend.c_str(),
            g_engine.requested_gpu_layers
        );
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
    jobjectArray roles,
    jobjectArray contents,
    jint max_tokens,
    jfloat temperature,
    jobject callback
) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    try {
        const auto request_started = std::chrono::steady_clock::now();
        if (!g_engine.model || !g_engine.ctx) {
            throw std::runtime_error("Model is not loaded");
        }
        const auto chat_roles = jobject_array_to_strings(env, roles);
        const auto chat_contents = jobject_array_to_strings(env, contents);
        const std::string prompt = format_chat_prompt(g_engine.model, chat_roles, chat_contents);
        if (!callback) {
            throw std::runtime_error("Missing streaming callback");
        }
        jclass callback_class = env->GetObjectClass(callback);
        if (!callback_class) {
            throw std::runtime_error("Unable to inspect streaming callback");
        }
        jmethodID on_token_method = env->GetMethodID(callback_class, "onToken", "([BIIIJJJJ)V");
        env->DeleteLocalRef(callback_class);
        if (!on_token_method) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            throw std::runtime_error("Streaming callback method was not found");
        }

        const StreamCallback stream_callback = [&](const InferenceResult & update) {
            auto output = env->NewByteArray(static_cast<jsize>(update.output.size()));
            if (!output) {
                if (env->ExceptionCheck()) env->ExceptionClear();
                throw std::runtime_error("Unable to allocate streaming output");
            }
            if (!update.output.empty()) {
                env->SetByteArrayRegion(
                    output,
                    0,
                    static_cast<jsize>(update.output.size()),
                    reinterpret_cast<const jbyte *>(update.output.data())
                );
            }
            env->CallVoidMethod(
                callback,
                on_token_method,
                output,
                static_cast<jint>(update.generated_tokens),
                static_cast<jint>(update.prompt_tokens),
                static_cast<jint>(update.cached_prompt_tokens),
                static_cast<jlong>(update.prompt_eval_ms),
                static_cast<jlong>(update.generation_ms),
                static_cast<jlong>(update.time_to_first_token_ms),
                static_cast<jlong>(update.total_ms)
            );
            env->DeleteLocalRef(output);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                throw std::runtime_error("Streaming callback failed");
            }
        };
        const auto result = run_inference(
            prompt,
            static_cast<int>(max_tokens),
            static_cast<float>(temperature),
            true,
            stream_callback,
            request_started
        );
        const std::string metrics =
            "prompt_tokens=" + std::to_string(result.prompt_tokens) +
            ";cached_prompt_tokens=" + std::to_string(result.cached_prompt_tokens) +
            ";generated_tokens=" + std::to_string(result.generated_tokens) +
            ";prompt_eval_ms=" + std::to_string(result.prompt_eval_ms) +
            ";generation_ms=" + std::to_string(result.generation_ms) +
            ";time_to_first_token_ms=" + std::to_string(result.time_to_first_token_ms) +
            ";total_ms=" + std::to_string(result.total_ms);
        return string_to_jstring(env, metrics);
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
        const std::vector<std::string> roles = {"user"};
        const std::vector<std::string> contents = {jstring_to_string(env, prompt)};
        const std::string formatted_prompt = format_chat_prompt(g_engine.model, roles, contents);
        g_engine.cached_tokens.clear();
        llama_memory_clear(llama_get_memory(g_engine.ctx), false);
        const auto inference = run_inference(
            formatted_prompt,
            static_cast<int>(max_tokens),
            static_cast<float>(temperature),
            false
        );
        const long long total_ms = inference.prompt_eval_ms + inference.generation_ms;
        const std::string result =
            "backend=" + g_engine.loaded_backend +
            ";build_type=" LLAMACPP_GEMMA_BUILD_TYPE +
            ";gpu_layers_requested=" + std::to_string(g_engine.requested_gpu_layers) +
            ";context_size=" + std::to_string(g_engine.requested_context_size) +
            ";threads=" + std::to_string(g_engine.requested_threads) +
            ";prompt_tokens=" + std::to_string(inference.prompt_tokens) +
            ";cached_prompt_tokens=" + std::to_string(inference.cached_prompt_tokens) +
            ";generated_tokens=" + std::to_string(inference.generated_tokens) +
            ";prompt_eval_ms=" + std::to_string(inference.prompt_eval_ms) +
            ";generation_ms=" + std::to_string(inference.generation_ms) +
            ";elapsed_ms=" + std::to_string(total_ms);

        g_engine.cached_tokens.clear();
        llama_memory_clear(llama_get_memory(g_engine.ctx), false);
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
