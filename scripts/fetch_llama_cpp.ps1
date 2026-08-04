$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $PSScriptRoot
$targetDir = Join-Path $rootDir "third_party\llama.cpp"
$revision = (Get-Content -Raw (Join-Path $PSScriptRoot "llama_cpp_revision.txt")).Trim()

if (Test-Path -LiteralPath (Join-Path $targetDir ".git")) {
    git -C $targetDir remote set-url origin https://github.com/ggml-org/llama.cpp.git
    if ($LASTEXITCODE -ne 0) { throw "Unable to update the llama.cpp remote" }
} else {
    if (Test-Path -LiteralPath $targetDir) {
        throw "$targetDir exists but is not a Git checkout"
    }
    New-Item -ItemType Directory -Path $targetDir | Out-Null
    git -C $targetDir init
    if ($LASTEXITCODE -ne 0) { throw "Unable to initialize $targetDir" }
    git -C $targetDir remote add origin https://github.com/ggml-org/llama.cpp.git
    if ($LASTEXITCODE -ne 0) { throw "Unable to add the llama.cpp remote" }
}

git -C $targetDir fetch --depth 1 origin $revision
if ($LASTEXITCODE -ne 0) { throw "Unable to fetch llama.cpp $revision" }
git -C $targetDir checkout --detach FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw "Unable to check out llama.cpp $revision" }

$actualRevision = (git -C $targetDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to read the checked-out llama.cpp revision" }
if ($actualRevision -ne $revision) {
    throw "Expected llama.cpp $revision but checked out $actualRevision"
}

Write-Host "llama.cpp $actualRevision is ready at $targetDir"
