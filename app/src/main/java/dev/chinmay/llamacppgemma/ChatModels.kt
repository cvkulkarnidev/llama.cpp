package dev.chinmay.llamacppgemma

import android.net.Uri

data class ChatMessage(
    val role: Role,
    val text: String,
) {
    enum class Role { User, Assistant, System }
}

data class InferenceSettings(
    val backend: LlamaBackend = LlamaBackend.Vulkan,
    val gpuLayers: Int = 99,
    val contextSize: Int = 2048,
    val threads: Int = 4,
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f,
)

data class BenchmarkResult(
    val backend: String,
    val buildType: String,
    val gpuLayersRequested: Int,
    val contextSize: Int,
    val threads: Int,
    val acceleratedDeviceCount: Int,
    val acceleratedDevices: String,
    val gpuOffloadDetected: Boolean,
    val offloadedLayers: Int,
    val promptTokens: Int,
    val generatedTokens: Int,
    val elapsedMs: Long,
) {
    val tokensPerSecond: Double
        get() = if (elapsedMs <= 0L) 0.0 else generatedTokens * 1000.0 / elapsedMs.toDouble()
}

data class ChatUiState(
    val modelUri: Uri? = null,
    val modelPath: String? = null,
    val modelName: String = "No model selected",
    val loadedBackend: String? = null,
    val nativeDiagnostics: String = "",
    val benchmark: BenchmarkResult? = null,
    val settings: InferenceSettings = InferenceSettings(),
    val messages: List<ChatMessage> = emptyList(),
    val input: String = "",
    val isBusy: Boolean = false,
    val error: String? = null,
)
