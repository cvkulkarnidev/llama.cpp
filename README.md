# LlamaCpp Gemma Android

Native Android chat app for running instruction-tuned GGUF models locally with `llama.cpp`.
The first device target is a Samsung Galaxy S24.

## Current backend support

| Backend | This APK | Notes |
| --- | --- | --- |
| ARM64 CPU | Yes | Explicit CPU-only device selection; safe fallback. |
| Vulkan GPU | Yes | Explicitly selects the Vulkan device and offloads the requested model layers. |
| Qualcomm Hexagon NPU | No | Upstream support is experimental and needs a separate Snapdragon/Hexagon toolchain build. |
| Exynos NPU | No | Upstream `llama.cpp` does not provide an Exynos NPU backend. |

The app now reports the selected Vulkan device, native build type, pinned llama.cpp revision,
requested GPU layers, and upstream offload logs. A backend label is therefore evidence of an
available selected device, rather than merely echoing the UI request.

## Identify the S24 variant

Galaxy S24 hardware differs by market and revision. Once USB debugging is enabled and the phone
is connected, run:

```powershell
adb shell getprop ro.product.model
adb shell getprop ro.soc.manufacturer
adb shell getprop ro.soc.model
adb shell getprop ro.hardware
```

- Exynos 2400 devices have an Xclipse 940 GPU and a Samsung NPU. Use this app's Vulkan path;
  the Samsung NPU is not exposed through a llama.cpp backend.
- Snapdragon devices can also use Vulkan in this APK. A Hexagon NPU build should be a separate
  product flavor because it requires the Qualcomm/Hexagon toolchain and additional runtime
  libraries.

Upstream's current Snapdragon instructions are in
[`docs/backend/snapdragon/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md).

## Recommended first model

Start with Google's instruction-tuned QAT Q4_0 GGUF:

```text
google/gemma-4-E2B-it-qat-q4_0-gguf
```

The app is text-only. Pick the main `.gguf` model file, not an `mmproj` or assistant/drafter file.
The selected model is copied to app-private storage so native llama.cpp receives a normal file
path. Keep several gigabytes of free storage available.

Recommended first settings:

- Backend: `Vulkan GPU`
- GPU layers: `99`
- Context: `2048`
- Threads: `4`

If model loading is unstable, try GPU layers `48`, `32`, then `16`; next reduce context to `1024`.
CPU mode is the compatibility fallback.

## Reproducible dependency setup

The exact upstream llama.cpp Git revision is stored in
[`scripts/llama_cpp_revision.txt`](scripts/llama_cpp_revision.txt). Fetch it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_llama_cpp.ps1
```

On Linux/macOS or in GitHub Actions:

```bash
bash scripts/fetch_llama_cpp.sh
```

Do not replace this with an unpinned `git pull`: llama.cpp's C API changes frequently.

## Build

The project uses JDK 17, Android SDK 35, NDK `27.2.12479018`, CMake `3.22.1`, and Gradle `8.11.1`.
The checked-in Gradle wrapper verifies the Gradle distribution SHA-256 before using it.

### GitHub Actions

Every push to `main` runs the `Android APK` workflow and uploads
`llamacpp-gemma-release-apk`. This is the easiest reproducible Vulkan build because the workflow
installs `glslc`, Vulkan headers, and SPIR-V headers before compiling.

Release APKs use the checked-in `app/dev-signing.p12` development key so successive CI builds can
update the same sideloaded app without deleting its private model files. The key and its public
password are intentionally reproducible and must not be used for a Play Store or production build.

### Local Android Studio

1. Install Android Studio with JDK 17, SDK 35, NDK `27.2.12479018`, and CMake `3.22.1`.
2. Install the LunarG Vulkan SDK so `glslc` and Vulkan/SPIR-V headers are available.
3. Run the pinned dependency fetch script above.
4. Open the repository root and build the `app` module.

If the Vulkan SDK is not auto-detected, set these environment variables before starting Android
Studio:

```text
SPIRV_HEADERS_DIR=<directory containing SPIRV-HeadersConfig.cmake>
SPIRV_HEADERS_INCLUDE_DIR=<directory containing spirv/unified1/spirv.hpp>
VULKAN_HEADERS_INCLUDE_DIR=<Vulkan-Headers include directory>
```

Command-line release build:

```powershell
.\gradlew.bat :app:assembleRelease
```

The APK is written to `app/build/outputs/apk/release/app-release.apk`.

## Device test sequence

1. Enable Developer options and USB debugging on the S24.
2. Connect USB and confirm `adb devices` reports `device`.
3. Install with `adb install -r app-release.apk`.
4. Copy/download the model to the phone, open the app, and tap `Pick GGUF`.
5. Load with Vulkan and open `Show settings`.
6. Confirm diagnostics show a `Vulkan` device and an upstream `offloaded ... layers` message.
7. Run `Benchmark`, then repeat in CPU mode for comparison.

The benchmark reports prompt-evaluation time separately from generation time. Displayed tokens/sec
uses generation time only.

## Implementation notes and limits

- Minimum SDK 29; target/compile SDK 35; ABI `arm64-v8a`.
- Kotlin, Jetpack Compose Material 3, MVVM/`StateFlow`, and a C++ JNI bridge.
- Chat messages are formatted with the GGUF's embedded chat template.
- Context overflow is checked and requested output is capped to available context space.
- Generation is currently non-streaming and one request runs at a time.
- Hard native driver crashes or process-level out-of-memory kills cannot be converted into a Kotlin
  error dialog.
- NPU via LiteRT-LM would require a second inference engine and a non-GGUF model deployment path;
  it is not a switch that can be enabled on this llama.cpp build.

## Primary references

- [llama.cpp Android documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [llama.cpp Vulkan build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan)
- [llama.cpp Snapdragon CPU/GPU/Hexagon backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
- [Android NDK Vulkan guidance](https://developer.android.com/ndk/guides/graphics/getting-started)
- [Samsung Exynos 2400 specifications](https://semiconductor.samsung.com/processor/mobile-processor/exynos-2400/)
- [Google Gemma 4 E2B Q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Android NNAPI deprecation and migration guidance](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
