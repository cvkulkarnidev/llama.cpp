package dev.chinmay.llamacppgemma

import android.net.Uri

data class ChatMessage(
    val role: Role,
    val text: String,
    val metrics: GenerationMetrics? = null,
) {
    enum class Role { User, Assistant, System }
}

data class GenerationMetrics(
    val promptTokens: Int,
    val cachedPromptTokens: Int,
    val generatedTokens: Int,
    val promptEvalMs: Long,
    val generationMs: Long,
    val timeToFirstTokenMs: Long,
    val requestMs: Long,
    val isComplete: Boolean,
) {
    val totalMs: Long
        get() = requestMs

    val tokensPerSecond: Double
        get() = if (generationMs <= 0L) 0.0 else generatedTokens * 1000.0 / generationMs.toDouble()
}

data class InferenceSettings(
    val backend: LlamaBackend = LlamaBackend.Vulkan,
    val gpuLayers: Int = 99,
    val contextSize: Int = 2048,
    val threads: Int = 4,
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f,
)

data class UiNotice(
    val id: Long,
    val message: String,
)

data class BenchmarkResult(
    val backend: String,
    val buildType: String,
    val gpuLayersRequested: Int,
    val contextSize: Int,
    val threads: Int,
    val promptTokens: Int,
    val cachedPromptTokens: Int,
    val generatedTokens: Int,
    val promptEvalMs: Long,
    val generationMs: Long,
) {
    val tokensPerSecond: Double
        get() = if (generationMs <= 0L) 0.0 else generatedTokens * 1000.0 / generationMs.toDouble()

    val totalMs: Long
        get() = promptEvalMs + generationMs
}

data class ChatUiState(
    val modelUri: Uri? = null,
    val modelPath: String? = null,
    val modelName: String = "No model selected",
    val loadedBackend: String? = null,
    val runtimeReport: String = "",
    val npuStatus: String = "Checking Android accelerator devices…",
    val nativeDiagnostics: String = "",
    val benchmark: BenchmarkResult? = null,
    val settings: InferenceSettings = InferenceSettings(),
    val messages: List<ChatMessage> = emptyList(),
    val input: String = "",
    val isBusy: Boolean = false,
    val error: String? = null,
    val notice: UiNotice? = null,
)
