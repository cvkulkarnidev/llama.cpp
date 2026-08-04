package dev.chinmay.llamacppgemma

enum class LlamaBackend(
    val label: String,
    val nativeName: String,
    val defaultGpuLayers: Int,
    val defaultContextSize: Int,
    val defaultThreads: Int,
) {
    Cpu("CPU safe mode", "cpu", 0, 1024, 4),
    Vulkan("Vulkan GPU", "vulkan", 99, 2048, 4),
}
