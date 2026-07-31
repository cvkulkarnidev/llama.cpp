package dev.chinmay.llamacppgemma

enum class LlamaBackend(
    val label: String,
    val nativeName: String,
    val defaultGpuLayers: Int,
) {
    Cpu("CPU", "cpu", 0),
    Vulkan("Vulkan GPU", "vulkan", 99),
    OpenCl("Adreno OpenCL GPU", "opencl", 99),
    Qnn("Experimental QNN/NPU", "qnn", 99),
}
