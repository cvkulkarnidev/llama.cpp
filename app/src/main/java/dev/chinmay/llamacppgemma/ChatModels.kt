package dev.chinmay.llamacppgemma

import android.net.Uri

data class ChatMessage(
    val role: Role,
    val text: String,
) {
    enum class Role { User, Assistant, System }
}

data class InferenceSettings(
    val backend: LlamaBackend = LlamaBackend.Cpu,
    val gpuLayers: Int = 0,
    val contextSize: Int = 4096,
    val threads: Int = maxOf(2, Runtime.getRuntime().availableProcessors() - 2),
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f,
)

data class ChatUiState(
    val modelUri: Uri? = null,
    val modelPath: String? = null,
    val modelName: String = "No model selected",
    val loadedBackend: String? = null,
    val settings: InferenceSettings = InferenceSettings(),
    val messages: List<ChatMessage> = emptyList(),
    val input: String = "",
    val isBusy: Boolean = false,
    val error: String? = null,
)
