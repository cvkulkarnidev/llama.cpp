package dev.chinmay.llamacppgemma

object LlamaBridge {
    init {
        System.loadLibrary("llamacpp_jni")
    }

    external fun loadModel(
        modelPath: String,
        backend: String,
        gpuLayers: Int,
        contextSize: Int,
        threads: Int,
    ): String

    external fun generate(
        roles: Array<String>,
        contents: Array<String>,
        maxTokens: Int,
        temperature: Float,
        callback: GenerationCallback,
    ): String

    external fun benchmark(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String

    external fun diagnostics(): String

    external fun runtimeReport(): String

    external fun acceleratorDevices(): String

    external fun unload()
}

fun interface GenerationCallback {
    fun onToken(
        outputUtf8: ByteArray,
        generatedTokens: Int,
        promptTokens: Int,
        cachedPromptTokens: Int,
        promptEvalMs: Long,
        generationMs: Long,
        timeToFirstTokenMs: Long,
        totalMs: Long,
    )
}
