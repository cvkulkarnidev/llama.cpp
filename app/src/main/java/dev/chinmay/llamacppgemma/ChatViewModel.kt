package dev.chinmay.llamacppgemma

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state
    private var nextNoticeId = 0L

    init {
        viewModelScope.launch {
            val status = runCatching {
                withContext(Dispatchers.IO) { LlamaBridge.acceleratorDevices() }
            }.fold(
                onSuccess = { inventory ->
                    "$inventory. Samsung ENN/NNAPI cannot execute GGUF through this llama.cpp build."
                },
                onFailure = { error -> "Accelerator probe failed: ${error.message}" },
            )
            Log.i(LOG_TAG, "NPU probe: $status")
            _state.update { it.copy(npuStatus = status) }
        }
        viewModelScope.launch {
            val existingModel = withContext(Dispatchers.IO) {
                File(getApplication<Application>().filesDir, "models")
                    .listFiles()
                    .orEmpty()
                    .filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }
                    .maxByOrNull(File::lastModified)
            }
            existingModel?.let { model ->
                _state.update { state ->
                    if (state.modelPath == null) {
                        state.copy(modelPath = model.absolutePath, modelName = model.name)
                    } else {
                        state
                    }
                }
            }
        }
    }

    fun setInput(value: String) {
        _state.update { it.copy(input = value) }
    }

    fun setBackend(backend: LlamaBackend) {
        val current = _state.value
        if (current.isBusy || current.settings.backend == backend) return
        _state.update {
            it.copy(
                settings = it.settings.copy(
                    backend = backend,
                    gpuLayers = backend.defaultGpuLayers,
                    contextSize = backend.defaultContextSize,
                    threads = backend.defaultThreads,
                ),
                loadedBackend = null,
                runtimeReport = "",
                benchmark = null,
            )
        }
    }

    fun setGpuLayers(value: Int) {
        val current = _state.value
        val minimum = if (current.settings.backend == LlamaBackend.Vulkan) 1 else 0
        val gpuLayers = value.coerceIn(minimum, 999)
        if (current.isBusy || current.settings.gpuLayers == gpuLayers) return
        _state.update {
            it.copy(
                settings = it.settings.copy(gpuLayers = gpuLayers),
                loadedBackend = null,
                runtimeReport = "",
                benchmark = null,
            )
        }
    }

    fun dismissError() {
        _state.update { it.copy(error = null) }
    }

    fun dismissNotice(id: Long) {
        _state.update { state ->
            if (state.notice?.id == id) state.copy(notice = null) else state
        }
    }

    fun selectModel(uri: Uri) {
        if (_state.value.isBusy) return
        viewModelScope.launch {
            _state.update {
                it.copy(
                    isBusy = true,
                    error = null,
                    loadedBackend = null,
                    runtimeReport = "",
                    benchmark = null,
                )
            }
            runCatching {
                withContext(Dispatchers.IO) {
                    LlamaBridge.unload()
                    copyModelToPrivateStorage(uri)
                }
            }.onSuccess { model ->
                _state.update {
                    it.copy(
                        modelUri = uri,
                        modelPath = model.absolutePath,
                        modelName = model.name,
                        loadedBackend = null,
                        runtimeReport = "",
                        nativeDiagnostics = "",
                        benchmark = null,
                        messages = emptyList(),
                        isBusy = false,
                    )
                }
            }.onFailure { e ->
                _state.update { it.copy(isBusy = false, error = e.message ?: "Unable to copy model") }
            }
        }
    }

    fun loadModel() {
        val current = _state.value
        val modelPath = current.modelPath ?: run {
            _state.update { it.copy(error = "Select a GGUF model first") }
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(isBusy = true, error = null) }
            runCatching {
                withContext(Dispatchers.IO) {
                    val backend = LlamaBridge.loadModel(
                        modelPath = modelPath,
                        backend = current.settings.backend.nativeName,
                        gpuLayers = current.settings.gpuLayers,
                        contextSize = current.settings.contextSize,
                        threads = current.settings.threads,
                    )
                    LoadedModel(
                        backend = backend,
                        runtimeReport = LlamaBridge.runtimeReport(),
                        diagnostics = LlamaBridge.diagnostics(),
                    )
                }
            }.onSuccess { loaded ->
                val toast = backendNotice(loaded.backend, loaded.runtimeReport)
                Log.i(LOG_TAG, "Model loaded: ${loaded.runtimeReport}")
                _state.update {
                    it.copy(
                        loadedBackend = loaded.backend,
                        runtimeReport = loaded.runtimeReport,
                        nativeDiagnostics = loaded.diagnostics,
                        benchmark = null,
                        isBusy = false,
                        notice = UiNotice(++nextNoticeId, toast),
                        messages = it.messages + ChatMessage(
                            ChatMessage.Role.System,
                            "Loaded ${it.modelName} on ${loaded.backend}",
                        ),
                    )
                }
            }.onFailure { e ->
                _state.update {
                    it.copy(
                        isBusy = false,
                        nativeDiagnostics = runCatching { LlamaBridge.diagnostics() }.getOrDefault(""),
                        error = e.message ?: "Unable to load model",
                    )
                }
            }
        }
    }

    fun benchmark() {
        val current = _state.value
        if (current.loadedBackend == null) {
            _state.update { it.copy(error = "Load the model before running benchmark") }
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(isBusy = true, error = null, benchmark = null) }
            runCatching {
                withContext(Dispatchers.Default) {
                    LlamaBridge.benchmark(
                        prompt = BENCHMARK_PROMPT,
                        maxTokens = 64,
                        temperature = 0.0f,
                    )
                }
            }.mapCatching { raw ->
                parseBenchmark(raw, current.loadedBackend.orEmpty())
            }.onSuccess { result ->
                Log.i(
                    LOG_TAG,
                    "Benchmark backend=${result.backend} tokens=${result.generatedTokens} " +
                        "generationMs=${result.generationMs} tokPerSec=${result.tokensPerSecond}",
                )
                _state.update {
                    it.copy(
                        isBusy = false,
                        benchmark = result,
                        nativeDiagnostics = LlamaBridge.diagnostics(),
                        messages = it.messages + ChatMessage(
                            ChatMessage.Role.System,
                            "Benchmark: ${"%.2f".format(result.tokensPerSecond)} tok/s (${result.buildType}, ${result.generatedTokens} tokens in ${result.generationMs} ms; prompt ${result.promptEvalMs} ms)",
                        ),
                    )
                }
            }.onFailure { e ->
                _state.update {
                    it.copy(
                        isBusy = false,
                        nativeDiagnostics = runCatching { LlamaBridge.diagnostics() }.getOrDefault(""),
                        error = e.message ?: "Benchmark failed",
                    )
                }
            }
        }
    }

    fun send() {
        val current = _state.value
        val userText = current.input.trim()
        if (userText.isEmpty() || current.isBusy) return
        if (current.loadedBackend == null) {
            _state.update { it.copy(error = "Load the model before chatting") }
            return
        }

        val chat = (current.messages
            .filter { it.role != ChatMessage.Role.System }
            .takeLast(8) + ChatMessage(ChatMessage.Role.User, userText))
        val roles = chat.map { message ->
            when (message.role) {
                ChatMessage.Role.User -> "user"
                ChatMessage.Role.Assistant -> "assistant"
                ChatMessage.Role.System -> "system"
            }
        }.toTypedArray()
        val contents = chat.map(ChatMessage::text).toTypedArray()
        _state.update {
            it.copy(
                input = "",
                isBusy = true,
                error = null,
                messages = it.messages + listOf(
                    ChatMessage(ChatMessage.Role.User, userText),
                    ChatMessage(
                        role = ChatMessage.Role.Assistant,
                        text = "",
                        metrics = GenerationMetrics(
                            promptTokens = 0,
                            cachedPromptTokens = 0,
                            generatedTokens = 0,
                            promptEvalMs = 0,
                            generationMs = 0,
                            timeToFirstTokenMs = 0,
                            requestMs = 0,
                            isComplete = false,
                        ),
                    ),
                ),
            )
        }

        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    val callback = GenerationCallback {
                            outputUtf8,
                            generatedTokens,
                            promptTokens,
                            cachedPromptTokens,
                            promptEvalMs,
                            generationMs,
                            timeToFirstTokenMs,
                            totalMs ->
                        val output = String(outputUtf8, Charsets.UTF_8)
                        _state.update { state ->
                            val messages = state.messages.toMutableList()
                            val index = messages.lastIndex
                            if (index >= 0 && messages[index].role == ChatMessage.Role.Assistant) {
                                messages[index] = messages[index].copy(
                                    text = output,
                                    metrics = GenerationMetrics(
                                        promptTokens = promptTokens,
                                        cachedPromptTokens = cachedPromptTokens,
                                        generatedTokens = generatedTokens,
                                        promptEvalMs = promptEvalMs,
                                        generationMs = generationMs,
                                        timeToFirstTokenMs = timeToFirstTokenMs,
                                        requestMs = totalMs,
                                        isComplete = false,
                                    ),
                                )
                            }
                            state.copy(messages = messages)
                        }
                    }
                    LlamaBridge.generate(
                        roles = roles,
                        contents = contents,
                        maxTokens = current.settings.maxTokens,
                        temperature = current.settings.temperature,
                        callback = callback,
                    )
                }
            }.mapCatching(::parseGenerationMetrics).onSuccess { metrics ->
                Log.i(
                    LOG_TAG,
                    "Generation backend=${current.loadedBackend} tokens=${metrics.generatedTokens} " +
                        "promptMs=${metrics.promptEvalMs} generationMs=${metrics.generationMs} " +
                        "tokPerSec=${metrics.tokensPerSecond} totalMs=${metrics.totalMs}",
                )
                _state.update { state ->
                    val messages = state.messages.toMutableList()
                    val index = messages.lastIndex
                    if (index >= 0 && messages[index].role == ChatMessage.Role.Assistant) {
                        messages[index] = messages[index].copy(
                            metrics = metrics,
                        )
                    }
                    state.copy(
                        isBusy = false,
                        nativeDiagnostics = runCatching { LlamaBridge.diagnostics() }.getOrDefault(""),
                        messages = messages,
                    )
                }
            }.onFailure { e ->
                _state.update { it.copy(isBusy = false, error = e.message ?: "Generation failed") }
            }
        }
    }

    override fun onCleared() {
        runCatching { LlamaBridge.unload() }
        super.onCleared()
    }

    private fun copyModelToPrivateStorage(uri: Uri): File {
        val context = getApplication<Application>()
        val modelsDir = File(context.filesDir, "models").apply { mkdirs() }
        val metadata = queryModelMetadata(uri)
        val displayName = metadata.name.ifBlank { "model.gguf" }
        require(displayName.endsWith(".gguf", ignoreCase = true)) { "Select a .gguf model file" }
        val outFile = File(modelsDir, displayName.replace(Regex("[^A-Za-z0-9._-]"), "_"))
        if (metadata.size != null && outFile.isFile && outFile.length() == metadata.size) {
            return outFile
        }

        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open selected model" }
            outFile.outputStream().use { output -> input.copyTo(output, MODEL_COPY_BUFFER_BYTES) }
        }
        return outFile
    }

    private fun queryModelMetadata(uri: Uri): ModelMetadata {
        val context = getApplication<Application>()
        return context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (!cursor.moveToFirst()) return@use ModelMetadata("", null)
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            ModelMetadata(
                name = if (nameIndex >= 0) cursor.getString(nameIndex).orEmpty() else "",
                size = if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) cursor.getLong(sizeIndex) else null,
            )
        } ?: ModelMetadata("", null)
    }

    private fun parseBenchmark(raw: String, fallbackBackend: String): BenchmarkResult {
        val values = raw.split(';')
            .mapNotNull { part ->
                val pieces = part.split('=', limit = 2)
                if (pieces.size == 2) pieces[0] to pieces[1] else null
            }
            .toMap()

        return BenchmarkResult(
            backend = values["backend"].orEmpty().ifBlank { fallbackBackend },
            buildType = values["build_type"].orEmpty().ifBlank { "Unknown" },
            gpuLayersRequested = values["gpu_layers_requested"]?.toIntOrNull() ?: 0,
            contextSize = values["context_size"]?.toIntOrNull() ?: 0,
            threads = values["threads"]?.toIntOrNull() ?: 0,
            promptTokens = values["prompt_tokens"]?.toIntOrNull() ?: 0,
            cachedPromptTokens = values["cached_prompt_tokens"]?.toIntOrNull() ?: 0,
            generatedTokens = values["generated_tokens"]?.toIntOrNull() ?: 0,
            promptEvalMs = values["prompt_eval_ms"]?.toLongOrNull() ?: 0L,
            generationMs = values["generation_ms"]?.toLongOrNull()
                ?: values["elapsed_ms"]?.toLongOrNull()
                ?: 0L,
        )
    }

    private fun parseGenerationMetrics(raw: String): GenerationMetrics {
        val values = raw.split(';')
            .mapNotNull { part ->
                val pieces = part.split('=', limit = 2)
                if (pieces.size == 2) pieces[0] to pieces[1] else null
            }
            .toMap()

        return GenerationMetrics(
            promptTokens = values["prompt_tokens"]?.toIntOrNull() ?: 0,
            cachedPromptTokens = values["cached_prompt_tokens"]?.toIntOrNull() ?: 0,
            generatedTokens = values["generated_tokens"]?.toIntOrNull() ?: 0,
            promptEvalMs = values["prompt_eval_ms"]?.toLongOrNull() ?: 0L,
            generationMs = values["generation_ms"]?.toLongOrNull() ?: 0L,
            timeToFirstTokenMs = values["time_to_first_token_ms"]?.toLongOrNull() ?: 0L,
            requestMs = values["total_ms"]?.toLongOrNull() ?: 0L,
            isComplete = true,
        )
    }

    private fun backendNotice(backend: String, runtimeReport: String): String = when {
        runtimeReport.contains("gpu_weights_verified=true") ->
            "Running on $backend — GPU model allocation verified"
        backend.contains("CPU", ignoreCase = true) -> "Running on CPU"
        else -> "Running on $backend — check runtime report for placement"
    }

    private companion object {
        const val LOG_TAG = "LlamaCppGemma"
        const val MODEL_COPY_BUFFER_BYTES = 1024 * 1024
        const val BENCHMARK_PROMPT =
            "Give a concise checklist for running a small language model locally on Android."
    }

    private data class LoadedModel(
        val backend: String,
        val runtimeReport: String,
        val diagnostics: String,
    )

    private data class ModelMetadata(val name: String, val size: Long?)
}
