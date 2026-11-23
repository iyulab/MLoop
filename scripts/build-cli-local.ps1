# build-cli-local.ps1
# Builds the MLoop CLI as a single file and copies mloop.exe to an output directory.
#
#   .\scripts\build-cli-local.ps1                       # -> <repo>\artifacts\cli-local
#   .\scripts\build-cli-local.ps1 -OutputDir <dir>       # any directory you choose
#   $env:MLOOP_LOCAL_BIN = "<dir>"; .\scripts\build-cli-local.ps1

param(
    [string]$OutputDir = $env:MLOOP_LOCAL_BIN
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CliProject = Join-Path $ProjectRoot "tools\MLoop.CLI\MLoop.CLI.csproj"
$PublishDir = Join-Path $ProjectRoot "tools\MLoop.CLI\bin\publish"
if (-not $OutputDir) { $OutputDir = Join-Path $ProjectRoot "artifacts\cli-local" }

Write-Host "=== MLoop CLI Local Build ===" -ForegroundColor Cyan
Write-Host "Project: $CliProject"
Write-Host "Output:  $OutputDir\mloop.exe"
Write-Host ""

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Yellow
}

# Clean previous publish
if (Test-Path $PublishDir) {
    Remove-Item -Recurse -Force $PublishDir
}

# Publish as single file
Write-Host "Building..." -ForegroundColor Yellow
dotnet publish $CliProject `
    -c Release `
    -r win-x64 `
    -o $PublishDir `
    --self-contained false `
    /p:PublishSingleFile=true `
    /p:IncludeNativeLibrariesForSelfExtract=true

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Copy to lib directory
$SourceExe = Join-Path $PublishDir "mloop.exe"
$DestExe = Join-Path $OutputDir "mloop.exe"

if (Test-Path $SourceExe) {
    Copy-Item -Path $SourceExe -Destination $DestExe -Force
    Write-Host ""
    Write-Host "SUCCESS: mloop.exe deployed to $DestExe" -ForegroundColor Green

    # Show version
    Write-Host ""
    & $DestExe --version
} else {
    Write-Host "ERROR: mloop.exe not found in publish output" -ForegroundColor Red
    exit 1
}
