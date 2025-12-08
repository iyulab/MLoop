#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Runs all tests including heavy tests (LLM, Slow, E2E, Database).

.DESCRIPTION
    This script runs the complete test suite locally, including tests that are
    excluded from CI/CD pipelines due to external dependencies or long execution times.

.PARAMETER Category
    Run only specific test category. Options: Unit, Integration, LLM, Slow, E2E, Database

.PARAMETER ExcludeHeavy
    Exclude heavy tests (same as CI). Runs only Unit and Integration tests.

.PARAMETER Coverage
    Enable code coverage collection.

.EXAMPLE
    ./scripts/test-all.ps1
    # Runs all tests

.EXAMPLE
    ./scripts/test-all.ps1 -ExcludeHeavy
    # Runs only lightweight tests (same as CI)

.EXAMPLE
    ./scripts/test-all.ps1 -Category LLM
    # Runs only LLM tests (requires API keys)
#>

param(
    [ValidateSet("Unit", "Integration", "LLM", "Slow", "E2E", "Database")]
    [string]$Category,

    [switch]$ExcludeHeavy,

    [switch]$Coverage
)

$ErrorActionPreference = "Stop"

Write-Host "MLoop Test Runner" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host ""

# Build filter
$filter = ""
if ($Category) {
    $filter = "--filter `"Category=$Category`""
    Write-Host "Running $Category tests only" -ForegroundColor Yellow
} elseif ($ExcludeHeavy) {
    $filter = "--filter `"Category!=LLM&Category!=Slow&Category!=E2E&Category!=Database`""
    Write-Host "Running lightweight tests only (CI mode)" -ForegroundColor Yellow
} else {
    Write-Host "Running ALL tests (including heavy tests)" -ForegroundColor Green
}

# Check for API keys if running LLM tests
if (-not $ExcludeHeavy -and (-not $Category -or $Category -eq "LLM")) {
    if (-not $env:OPENAI_API_KEY -and -not $env:AZURE_OPENAI_API_KEY) {
        Write-Host ""
        Write-Host "Warning: No LLM API keys found. LLM tests may be skipped." -ForegroundColor Yellow
        Write-Host "Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable for LLM tests." -ForegroundColor Yellow
        Write-Host ""
    }
}

# Build coverage option
$coverageOption = ""
if ($Coverage) {
    $coverageOption = '--collect:"XPlat Code Coverage"'
    Write-Host "Code coverage enabled" -ForegroundColor Cyan
}

Write-Host ""

# Run tests
$testCommand = "dotnet test MLoop.sln --configuration Release --verbosity normal $filter $coverageOption"
Write-Host "Executing: $testCommand" -ForegroundColor DarkGray
Write-Host ""

Invoke-Expression $testCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "All tests passed!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Some tests failed. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
