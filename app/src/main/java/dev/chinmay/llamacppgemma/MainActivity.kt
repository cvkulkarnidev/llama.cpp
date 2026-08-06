package dev.chinmay.llamacppgemma

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
    private val viewModel by viewModels<ChatViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val state by viewModel.state.collectAsStateWithLifecycle()
                ChatScreen(
                    state = state,
                    actions = ChatActions(
                        onModelSelected = viewModel::selectModel,
                        onLoad = viewModel::loadModel,
                        onSend = viewModel::send,
                        onInput = viewModel::setInput,
                        onBackend = viewModel::setBackend,
                        onGpuLayers = viewModel::setGpuLayers,
                        onBenchmark = viewModel::benchmark,
                        onDismissError = viewModel::dismissError,
                    ),
                )
            }
        }
    }
}

data class ChatActions(
    val onModelSelected: (android.net.Uri) -> Unit,
    val onLoad: () -> Unit,
    val onSend: () -> Unit,
    val onInput: (String) -> Unit,
    val onBackend: (LlamaBackend) -> Unit,
    val onGpuLayers: (Int) -> Unit,
    val onBenchmark: () -> Unit,
    val onDismissError: () -> Unit,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(state: ChatUiState, actions: ChatActions) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(actions.onModelSelected)
    }

    state.error?.let { error ->
        AlertDialog(
            onDismissRequest = actions.onDismissError,
            title = { Text("Runtime error") },
            text = { Text(error) },
            confirmButton = {
                TextButton(onClick = actions.onDismissError) {
                    Text("OK")
                }
            },
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Gemma 4 E2B llama.cpp") })
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .background(Color(0xFFFAFAFA))
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            ModelPanel(
                state = state,
                onPick = { picker.launch(arrayOf("*/*")) },
                onLoad = actions.onLoad,
                onBackend = actions.onBackend,
                onGpuLayers = actions.onGpuLayers,
                onBenchmark = actions.onBenchmark,
            )

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(state.messages) { message ->
                    MessageBubble(message)
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = state.input,
                    onValueChange = actions.onInput,
                    modifier = Modifier.weight(1f),
                    minLines = 1,
                    maxLines = 4,
                    placeholder = { Text("Ask locally...") },
                )
                Button(
                    enabled = !state.isBusy,
                    onClick = actions.onSend,
                    modifier = Modifier.align(Alignment.CenterVertically),
                ) {
                    Text(if (state.isBusy) "Wait" else "Send")
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelPanel(
    state: ChatUiState,
    onPick: () -> Unit,
    onLoad: () -> Unit,
    onBackend: (LlamaBackend) -> Unit,
    onGpuLayers: (Int) -> Unit,
    onBenchmark: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }

    Card(shape = RoundedCornerShape(8.dp)) {
        Column(
            modifier = Modifier.padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(state.modelName, style = MaterialTheme.typography.titleSmall)
                    Text(
                        runtimeStatusText(state.loadedBackend),
                        style = MaterialTheme.typography.bodySmall,
                        color = runtimeStatusColor(state.loadedBackend),
                    )
                }
                TextButton(onClick = onPick) { Text("Pick GGUF") }
                Button(enabled = !state.isBusy, onClick = onLoad) { Text("Load") }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    state.benchmark?.let { result ->
                        Text(
                            if (result.gpuOffloadDetected) "ON GPU" else "NOT ON GPU",
                            style = MaterialTheme.typography.bodySmall,
                            color = if (result.gpuOffloadDetected) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.error
                            },
                        )
                        Text(
                            "Decode: ${"%.2f".format(result.decodeTokensPerSecond)} tok/s, total: ${"%.2f".format(result.tokensPerSecond)} tok/s",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            "Prompt: ${result.promptTokens} tokens in ${result.promptElapsedMs} ms (${"%.2f".format(result.promptTokensPerSecond)} tok/s)",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            "${result.generatedTokens} generated tokens in ${result.generationElapsedMs} ms, ${result.buildType}, ${result.backend}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            "gpuLayers=${result.gpuLayersRequested}, ctx=${result.contextSize}, threads=${result.threads}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            "gpuDevices=${result.acceleratedDeviceCount}, offload=${if (result.gpuOffloadDetected) "yes" else "no"}, offloadedLayers=${result.offloadedLayers}",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (result.gpuOffloadDetected) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.error
                            },
                        )
                        if (result.acceleratedDevices.isNotBlank()) {
                            Text(
                                result.acceleratedDevices,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    } ?: Text(
                        "S24 Exynos: use Vulkan first; CPU is safe fallback.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                TextButton(
                    enabled = !state.isBusy && state.loadedBackend != null,
                    onClick = onBenchmark,
                ) {
                    Text("Benchmark")
                }
            }

            TextButton(onClick = { showSettings = !showSettings }) {
                Text(if (showSettings) "Hide settings" else "Show settings")
            }

            if (showSettings) {
                ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
                    OutlinedTextField(
                        value = state.settings.backend.label,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Backend") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        modifier = Modifier
                            .menuAnchor()
                            .fillMaxWidth(),
                    )
                    ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        LlamaBackend.entries.forEach { backend ->
                            DropdownMenuItem(
                                text = { Text(backend.label) },
                                onClick = {
                                    expanded = false
                                    onBackend(backend)
                                },
                            )
                        }
                    }
                }

                Text("GPU layers: ${state.settings.gpuLayers}")
                Slider(
                    value = state.settings.gpuLayers.toFloat(),
                    onValueChange = { onGpuLayers(it.toInt()) },
                    valueRange = 0f..99f,
                    steps = 98,
                )

                if (state.nativeDiagnostics.isNotBlank()) {
                    Text("Native diagnostics", style = MaterialTheme.typography.titleSmall)
                    Text(
                        state.nativeDiagnostics.trim().takeLast(1600),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun runtimeStatusText(loadedBackend: String?): String {
    return when {
        loadedBackend == null -> "Model not loaded"
        loadedBackend.contains("Vulkan GPU active", ignoreCase = true) -> "ON GPU - $loadedBackend"
        else -> "NOT ON GPU - $loadedBackend"
    }
}

@Composable
private fun runtimeStatusColor(loadedBackend: String?): Color {
    return when {
        loadedBackend == null -> MaterialTheme.colorScheme.onSurfaceVariant
        loadedBackend.contains("Vulkan GPU active", ignoreCase = true) -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.error
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val isUser = message.role == ChatMessage.Role.User
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            shape = RoundedCornerShape(8.dp),
            color = when (message.role) {
                ChatMessage.Role.User -> MaterialTheme.colorScheme.primaryContainer
                ChatMessage.Role.Assistant -> Color.White
                ChatMessage.Role.System -> Color(0xFFEFEFEF)
            },
            tonalElevation = 1.dp,
            modifier = Modifier.fillMaxWidth(if (message.role == ChatMessage.Role.System) 1f else 0.88f),
        ) {
            Text(
                text = message.text,
                modifier = Modifier.padding(12.dp),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
    Spacer(Modifier.height(2.dp))
}
