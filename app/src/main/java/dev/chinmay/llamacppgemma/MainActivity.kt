package dev.chinmay.llamacppgemma

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
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
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
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
                        onDismissNotice = viewModel::dismissNotice,
                    ),
                )
            }
        }
    }
}

data class ChatActions(
    val onModelSelected: (Uri) -> Unit,
    val onLoad: () -> Unit,
    val onSend: () -> Unit,
    val onInput: (String) -> Unit,
    val onBackend: (LlamaBackend) -> Unit,
    val onGpuLayers: (Int) -> Unit,
    val onBenchmark: () -> Unit,
    val onDismissError: () -> Unit,
    val onDismissNotice: (Long) -> Unit,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(state: ChatUiState, actions: ChatActions) {
    val context = LocalContext.current
    var showSettings by rememberSaveable { mutableStateOf(false) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(actions.onModelSelected)
    }

    LaunchedEffect(state.notice?.id) {
        state.notice?.let { notice ->
            Toast.makeText(context, notice.message, Toast.LENGTH_LONG).show()
            actions.onDismissNotice(notice.id)
        }
    }

    state.error?.let { error ->
        AlertDialog(
            onDismissRequest = actions.onDismissError,
            title = { Text("Runtime error") },
            text = { Text(error) },
            confirmButton = {
                TextButton(onClick = actions.onDismissError) { Text("OK") }
            },
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (showSettings) "Settings" else "Gemma llama.cpp") },
                actions = {
                    TextButton(onClick = { showSettings = !showSettings }) {
                        Text(if (showSettings) "Done" else "Settings")
                    }
                },
            )
        },
    ) { padding ->
        if (showSettings) {
            SettingsScreen(
                state = state,
                actions = actions,
                onPick = { picker.launch(arrayOf("application/octet-stream", "*/*")) },
                modifier = Modifier.padding(padding),
            )
        } else {
            ChatContent(
                state = state,
                actions = actions,
                modifier = Modifier.padding(padding),
            )
        }
    }
}

@Composable
private fun ChatContent(
    state: ChatUiState,
    actions: ChatActions,
    modifier: Modifier = Modifier,
) {
    val messageListState = rememberLazyListState()
    LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.text?.length) {
        if (state.messages.isNotEmpty()) messageListState.scrollToItem(state.messages.lastIndex)
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFFFAFAFA))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        RuntimeStatusCard(state)

        LazyColumn(
            state = messageListState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(state.messages) { message -> MessageBubble(message) }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = state.input,
                onValueChange = actions.onInput,
                modifier = Modifier.weight(1f),
                enabled = !state.isBusy,
                minLines = 1,
                maxLines = 4,
                placeholder = { Text("Ask locally…") },
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

@Composable
private fun RuntimeStatusCard(state: ChatUiState) {
    Card(shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(state.modelName, style = MaterialTheme.typography.titleSmall)
            Text(
                state.loadedBackend?.let { "Running on $it" }
                    ?: "Model not loaded — open Settings",
                style = MaterialTheme.typography.bodySmall,
                color = if (state.loadedBackend != null) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
            state.benchmark?.let { result ->
                Text(
                    "${"%.2f".format(result.tokensPerSecond)} tok/s • " +
                        "${result.generatedTokens} tokens/${result.generationMs} ms",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsScreen(
    state: ChatUiState,
    actions: ChatActions,
    onPick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var backendMenuExpanded by remember { mutableStateOf(false) }
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("Model", style = MaterialTheme.typography.titleMedium)
                Text(state.modelName, style = MaterialTheme.typography.bodyMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(enabled = !state.isBusy, onClick = onPick) { Text("Pick GGUF") }
                    Button(
                        enabled = !state.isBusy && state.modelPath != null,
                        onClick = actions.onLoad,
                    ) {
                        Text(if (state.isBusy) "Loading…" else "Load model")
                    }
                }
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("Execution", style = MaterialTheme.typography.titleMedium)
                ExposedDropdownMenuBox(
                    expanded = backendMenuExpanded,
                    onExpandedChange = { if (!state.isBusy) backendMenuExpanded = it },
                ) {
                    OutlinedTextField(
                        value = state.settings.backend.label,
                        onValueChange = {},
                        enabled = !state.isBusy,
                        readOnly = true,
                        label = { Text("Backend") },
                        trailingIcon = {
                            ExposedDropdownMenuDefaults.TrailingIcon(expanded = backendMenuExpanded)
                        },
                        modifier = Modifier
                            .menuAnchor()
                            .fillMaxWidth(),
                    )
                    ExposedDropdownMenu(
                        expanded = backendMenuExpanded,
                        onDismissRequest = { backendMenuExpanded = false },
                    ) {
                        LlamaBackend.entries.forEach { backend ->
                            DropdownMenuItem(
                                text = { Text(backend.label) },
                                onClick = {
                                    backendMenuExpanded = false
                                    actions.onBackend(backend)
                                },
                            )
                        }
                    }
                }

                Text("GPU layers: ${state.settings.gpuLayers}")
                Slider(
                    value = state.settings.gpuLayers.toFloat(),
                    onValueChange = { actions.onGpuLayers(it.toInt()) },
                    enabled = !state.isBusy && state.settings.backend == LlamaBackend.Vulkan,
                    valueRange = if (state.settings.backend == LlamaBackend.Vulkan) 1f..99f else 0f..99f,
                    steps = if (state.settings.backend == LlamaBackend.Vulkan) 97 else 98,
                )

                Text(
                    state.loadedBackend?.let { "Active: $it" } ?: "Load the model to verify placement",
                    style = MaterialTheme.typography.bodySmall,
                )
                if (state.runtimeReport.isNotBlank()) {
                    RuntimePlacementReport(state.runtimeReport)
                }
                Button(
                    enabled = !state.isBusy && state.loadedBackend != null,
                    onClick = actions.onBenchmark,
                ) {
                    Text("Run 64-token benchmark")
                }
                state.benchmark?.let { BenchmarkDetails(it) }
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text("NPU probe", style = MaterialTheme.typography.titleMedium)
                Text(state.npuStatus, style = MaterialTheme.typography.bodySmall)
                Text(
                    "An NPU driver being present does not mean a GGUF model ran there. " +
                        "Only verified model allocation is reported as active.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        if (state.nativeDiagnostics.isNotBlank()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text("Native diagnostics", style = MaterialTheme.typography.titleMedium)
                    Text(
                        state.nativeDiagnostics.trim().takeLast(3000),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun RuntimePlacementReport(report: String) {
    val fields = remember(report) { parseReport(report) }
    val verified = fields["gpu_weights_verified"] == "true"
    Text(
        if (verified) "GPU allocation verified" else "CPU/host allocation",
        style = MaterialTheme.typography.titleSmall,
        color = if (verified) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
    )
    Text(
        "Model devices: ${fields["model_devices"].orEmpty()}\n" +
            "Model memory: GPU ${fields["model_gpu_mib"] ?: "0"} MiB, " +
            "CPU ${fields["model_cpu_mib"] ?: "0"} MiB\n" +
            "Context: GPU ${fields["context_gpu_mib"] ?: "0"} MiB, " +
            "CPU ${fields["context_cpu_mib"] ?: "0"} MiB\n" +
            "Compute: GPU ${fields["compute_gpu_mib"] ?: "0"} MiB, " +
            "CPU ${fields["compute_cpu_mib"] ?: "0"} MiB",
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun BenchmarkDetails(result: BenchmarkResult) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            "${"%.2f".format(result.tokensPerSecond)} tok/s",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.primary,
        )
        Text(
            "${result.generatedTokens} tokens/${result.generationMs} ms; " +
                "prompt ${result.promptTokens}/${result.promptEvalMs} ms; " +
                "GPU layers ${result.gpuLayersRequested}; ctx ${result.contextSize}; " +
                "threads ${result.threads}; ${result.buildType}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun parseReport(report: String): Map<String, String> = report
    .split(';')
    .mapNotNull { part ->
        val pieces = part.split('=', limit = 2)
        if (pieces.size == 2) pieces[0] to pieces[1] else null
    }
    .toMap()

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
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = if (message.text.isEmpty() && message.metrics?.isComplete == false) {
                        "…"
                    } else {
                        message.text.trim()
                    },
                    style = MaterialTheme.typography.bodyMedium,
                )
                message.metrics?.let { metrics ->
                    val timing = if (metrics.isComplete) {
                        "${metrics.generatedTokens} tokens • ${"%.2f".format(metrics.tokensPerSecond)} tok/s • " +
                            "TTFT ${metrics.timeToFirstTokenMs} ms • " +
                            "prompt ${metrics.promptTokens} (${metrics.cachedPromptTokens} cached)/${metrics.promptEvalMs} ms • " +
                            "generation ${metrics.generationMs} ms • total ${metrics.totalMs} ms"
                    } else {
                        "${metrics.generatedTokens} tokens • ${metrics.generationMs} ms • " +
                            "TTFT ${metrics.timeToFirstTokenMs} ms • " +
                            "prompt ${metrics.promptTokens} (${metrics.cachedPromptTokens} cached)"
                    }
                    Text(
                        text = timing,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
    Spacer(Modifier.height(2.dp))
}
