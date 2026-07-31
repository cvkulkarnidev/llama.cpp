# LlamaCpp Gemma Android

Android Studio starter app for running **Gemma 4 E2B GGUF** locally through `llama.cpp`.

The app is intentionally backend-pluggable:

- CPU works on normal Android ARM64 builds.
- GPU can work when `llama.cpp` is built with Vulkan or OpenCL support and the device driver is compatible.
- NPU is experimental and mostly Snapdragon/QNN/Hexagon-specific. It is not a generic Android feature.

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

Recommended first test:

- `Q4_K_M` or `Q4_0` for RAM and speed.
- Increase quality later if the phone has enough memory.

## Prepare llama.cpp

From the project root:

```bash
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
```

Then open this folder in Android Studio and build the `app` module.

If you prefer command line builds, install Gradle 8.11.1 or generate a wrapper:

```bash
gradle wrapper --gradle-version 8.11.1
./gradlew :app:assembleDebug
```

## CPU Build

CPU is the safest first build:

Use Android Studio, or run `./gradlew :app:assembleDebug` after generating the wrapper.

## Vulkan GPU Build

Enable Vulkan in `app/build.gradle.kts`:

```kotlin
arguments += listOf("-DGGML_VULKAN=ON")
```

Then in the app choose `Vulkan GPU` and keep `GPU layers` high, for example `99`.

## Qualcomm Adreno OpenCL GPU Build

Enable OpenCL:

```kotlin
arguments += listOf("-DGGML_OPENCL=ON")
```

OpenCL in llama.cpp targets Qualcomm Adreno first. It is verified only on newer Snapdragon/Adreno combinations, so older phones may still fall back or perform poorly.

## Snapdragon NPU / QNN

For NPU, do not expect a generic switch to work on every phone. You need a llama.cpp build that includes QNN/Hexagon support and the required Qualcomm runtime files for the target device.

This app exposes an `Experimental QNN/NPU` backend option, but the native library must actually be built with that backend. Treat it as a build flavor we can wire after confirming your exact phone SoC.

## Android Notes

- Minimum SDK: 29
- Target SDK: 35
- ABI: `arm64-v8a`
- UI: Jetpack Compose + Material 3
- Architecture: simple MVVM with `StateFlow`

## Practical Recommendation

For your current goal, I would test in this order:

1. CPU with `Q4_K_M`.
2. Vulkan GPU.
3. OpenCL if the phone has Snapdragon/Adreno.
4. QNN/NPU only if the device is Snapdragon and you are ready to include Qualcomm SDK/runtime setup.

If your phone is Samsung Galaxy S24 India/Exynos, QNN/NPU is not the right path. Vulkan is the realistic llama.cpp acceleration path; LiteRT/MediaPipe may still use GPU better than llama.cpp for some models.
