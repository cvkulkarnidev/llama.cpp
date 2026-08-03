plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val llamaCppRevision = rootProject.file("scripts/llama_cpp_revision.txt").readText().trim()

android {
    namespace = "dev.chinmay.llamacppgemma"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "dev.chinmay.llamacppgemma"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
                val cmakeArguments = mutableListOf(
                    "-DANDROID_STL=c++_shared",
                    "-DLLAMA_CPP_DIR=${rootDir}/third_party/llama.cpp",
                    "-DLLAMA_CPP_REVISION=$llamaCppRevision",
                    "-DGGML_VULKAN=ON",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_OPENCL=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                )
                System.getenv("SPIRV_HEADERS_DIR")?.takeIf { it.isNotBlank() }?.let {
                    cmakeArguments += "-DSPIRV-Headers_DIR=$it"
                }
                System.getenv("SPIRV_HEADERS_INCLUDE_DIR")?.takeIf { it.isNotBlank() }?.let {
                    cmakeArguments += "-DSPIRV_HEADERS_INCLUDE_DIR=$it"
                }
                System.getenv("VULKAN_HEADERS_INCLUDE_DIR")?.takeIf { it.isNotBlank() }?.let {
                    cmakeArguments += "-DVULKAN_HEADERS_INCLUDE_DIR=$it"
                }
                arguments += cmakeArguments
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
            isDebuggable = false
            isMinifyEnabled = false
        }
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
