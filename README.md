# LlamaCpp Gemma Android

Android Studio starter app for running **Gemma 4 E2B GGUF** locally through `llama.cpp`.

The app is configured for the Samsung S24 Exynos path:

- CPU works on normal Android ARM64 builds.
- Vulkan GPU is enabled in the Android native build with `-DGGML_VULKAN=ON`.
- NPU/QNN is not exposed for Exynos because it is mostly Snapdragon/QNN/Hexagon-specific.

## Best Existing Apps To Try First

If you want something installable before building your own app, try:

- **PocketPal AI**: GGUF/llama.cpp Android app.
- **ChatterUI**: Android chat UI with local model support.
- **SmolChat-Android**: lightweight llama.cpp Android app.
- **Maid**: Flutter app for GGUF/llama.cpp models.

Those are good references. This project is for your own Kotlin/Compose app where we can control backend flags, model path handling, and UI.

## Model

Use the official GGUF model:

```bash
llama-cli -hf ggml-org/gemma-4-E2B-it-GGUF --prompt "Hello"
```

For the Android app, download a `.gguf` file manually and select it inside the app. The app copies it into private app storage because native `llama.cpp` needs a normal filesystem path.

Recommended first test on S24 Exynos:

- `Q4_K_M` or `Q4_0` for RAM and speed.
- Backend: `Vulkan GPU`
- GPU layers: `99`
- Context size: `2048`
- Threads: `4`
- Increase quality or context later if the phone stays stable.

The app includes a **Benchmark** button after the model is loaded. It runs a fixed 64-token native benchmark and reports generated tokens per second, generated token count, elapsed time, and backend label.

For speed testing, use the **release APK** from GitHub Actions. The release build compiles native llama.cpp with `CMAKE_BUILD_TYPE=Release`; the earlier debug APK used `Debug`, which can be much slower. The release variant is signed with the debug signing key so it can be installed directly for local testing.

Tap **Show settings** after loading a model to see native diagnostics. The diagnostics include:

- native build type: `Release` or `Debug`
- backend request
- requested GPU layers, context size, and threads
- llama.cpp native logs, including Vulkan/offload messages when llama.cpp emits them

## Prepare llama.cpp

From the project root:

```bash
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
```

Then open this folder in Android Studio and build the `app` module.

If you prefer command line builds, install Gradle 8.11.1 or generate a wrapper:

```bash
gradle wrapper --gradle-version 8.11.1
./gradlew :app:assembleRelease
```

## CPU Build

CPU is the safest first build:

Use Android Studio, or run `./gradlew :app:assembleDebug` after generating the wrapper. This is useful for development, but it is not the build to use for measuring tokens/second.

## Vulkan GPU Build

Vulkan is already enabled in `app/build.gradle.kts`:

```kotlin
arguments += listOf(
    "-DGGML_VULKAN=ON",
    "-DGGML_OPENMP=OFF",
    "-DGGML_OPENCL=OFF",
)
```

In the app choose `Vulkan GPU` and keep `GPU layers` high, for example `99`.

## Snapdragon NPU / QNN

For NPU, do not expect a generic switch to work on every phone. You need a llama.cpp build that includes QNN/Hexagon support and the required Qualcomm runtime files for the target device.

This Exynos build does not expose an `Experimental QNN/NPU` backend option. Treat NPU as a separate Snapdragon-specific build flavor, not the default Android path.

## Android Notes

- Minimum SDK: 29
- Target SDK: 35
- ABI: `arm64-v8a`
- UI: Jetpack Compose + Material 3
- Architecture: simple MVVM with `StateFlow`
- Runtime errors are shown in a popup dialog. Hard native crashes such as process-level Vulkan driver aborts or out-of-memory kills can still terminate the app before Kotlin can display an error.
- The settings section is collapsed by default. Tap **Show settings** to change backend or GPU layers.

## Practical Recommendation

For your current goal, I would test in this order:

1. Vulkan GPU with `Q4_K_M`, `gpuLayers=99`, `contextSize=2048`, and `threads=4`.
2. If Vulkan crashes or falls back, reduce `gpuLayers` to `48`, then `32`, then `16`.
3. If memory is still unstable, reduce context size to `1024`.
4. CPU is the fallback path only.

If your phone is Samsung Galaxy S24 India/Exynos, QNN/NPU is not the right path. Vulkan is the realistic llama.cpp acceleration path; LiteRT/MediaPipe may still use GPU better than llama.cpp for some models.
