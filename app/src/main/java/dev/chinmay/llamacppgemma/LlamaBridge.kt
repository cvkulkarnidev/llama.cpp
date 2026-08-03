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
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String

    external fun benchmark(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String

    external fun diagnostics(): String

    external fun unload()
}
