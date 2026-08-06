package dev.chinmay.llamacppgemma

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
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

    fun setInput(value: String) {
        _state.update { it.copy(input = value) }
    }

    fun setBackend(backend: LlamaBackend) {
        _state.update {
            it.copy(
                settings = it.settings.copy(
                    backend = backend,
                    gpuLayers = backend.defaultGpuLayers,
                    contextSize = backend.defaultContextSize,
                    threads = backend.defaultThreads,
                ),
            )
        }
    }

    fun setGpuLayers(value: Int) {
        _state.update { it.copy(settings = it.settings.copy(gpuLayers = value.coerceIn(0, 999))) }
    }

    fun dismissError() {
        _state.update { it.copy(error = null) }
    }

    fun selectModel(uri: Uri) {
        viewModelScope.launch {
            _state.update { it.copy(isBusy = true, error = null) }
            runCatching {
                withContext(Dispatchers.IO) {
                    copyModelToPrivateStorage(uri)
                }
            }.onSuccess { model ->
                _state.update {
                    it.copy(
                        modelUri = uri,
                        modelPath = model.absolutePath,
                        modelName = model.name,
                        loadedBackend = null,
                        nativeDiagnostics = "",
                        benchmark = null,
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
                    LlamaBridge.loadModel(
                        modelPath = modelPath,
                        backend = current.settings.backend.nativeName,
                        gpuLayers = current.settings.gpuLayers,
                        contextSize = current.settings.contextSize,
                        threads = current.settings.threads,
                    )
                }
            }.onSuccess { backend ->
                _state.update {
                    it.copy(
                        loadedBackend = backend,
                        nativeDiagnostics = LlamaBridge.diagnostics(),
                        benchmark = null,
                        isBusy = false,
                        messages = it.messages + ChatMessage(
                            ChatMessage.Role.System,
                            "Loaded ${it.modelName} on $backend",
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
                _state.update {
                    it.copy(
                        isBusy = false,
                        benchmark = result,
                        nativeDiagnostics = LlamaBridge.diagnostics(),
                        messages = it.messages + ChatMessage(
                            ChatMessage.Role.System,
                            "Benchmark: ${result.gpuStatusLabel}, decode=${"%.2f".format(result.decodeTokensPerSecond)} tok/s, total=${"%.2f".format(result.tokensPerSecond)} tok/s (${result.generatedTokens} tokens)",
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

        val prompt = buildPrompt(current.messages, userText)
        _state.update {
            it.copy(
                input = "",
                isBusy = true,
                error = null,
                messages = it.messages + ChatMessage(ChatMessage.Role.User, userText),
            )
        }

        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    LlamaBridge.generate(
                        prompt = prompt,
                        maxTokens = current.settings.maxTokens,
                        temperature = current.settings.temperature,
                    )
                }
            }.onSuccess { answer ->
                _state.update {
                    it.copy(
                        isBusy = false,
                        messages = it.messages + ChatMessage(ChatMessage.Role.Assistant, answer.trim()),
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
        val displayName = queryDisplayName(uri).ifBlank { "model.gguf" }
        val outFile = File(modelsDir, displayName.replace(Regex("[^A-Za-z0-9._-]"), "_"))

        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open selected model" }
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }

    private fun queryDisplayName(uri: Uri): String {
        val context = getApplication<Application>()
        return context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) cursor.getString(index) else ""
        }.orEmpty()
    }

    private fun buildPrompt(history: List<ChatMessage>, userText: String): String {
        val turns = history
            .filter { it.role != ChatMessage.Role.System }
            .takeLast(8)
            .joinToString("\n") { message ->
                when (message.role) {
                    ChatMessage.Role.User -> "User: ${message.text}"
                    ChatMessage.Role.Assistant -> "Assistant: ${message.text}"
                    ChatMessage.Role.System -> ""
                }
            }

        return """
            <start_of_turn>user
            $turns
            User: $userText
            <end_of_turn>
            <start_of_turn>model
        """.trimIndent()
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
            acceleratedDeviceCount = values["accelerated_device_count"]?.toIntOrNull() ?: 0,
            acceleratedDevices = values["accelerated_devices"].orEmpty(),
            gpuOffloadDetected = values["gpu_offload_detected"]?.toBooleanStrictOrNull() ?: false,
            offloadedLayers = values["offloaded_layers"]?.toIntOrNull() ?: 0,
            promptTokens = values["prompt_tokens"]?.toIntOrNull() ?: 0,
            generatedTokens = values["generated_tokens"]?.toIntOrNull() ?: 0,
            promptElapsedMs = values["prompt_elapsed_ms"]?.toLongOrNull() ?: 0L,
            generationElapsedMs = values["generation_elapsed_ms"]?.toLongOrNull() ?: 0L,
            elapsedMs = values["elapsed_ms"]?.toLongOrNull() ?: 0L,
        )
    }

    private companion object {
        const val BENCHMARK_PROMPT = """
            <start_of_turn>user
            Give a concise checklist for running a small language model locally on Android.
            <end_of_turn>
            <start_of_turn>model
        """
    }
}

private val BenchmarkResult.gpuStatusLabel: String
    get() = if (gpuOffloadDetected) "ON GPU" else "NOT ON GPU"
