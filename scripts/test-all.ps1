#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Runs every test project, one at a time.

.DESCRIPTION
    Serially, never two at once: several suites run real trainer searches on a ten-second budget, and
    under CPU load no trial completes — a failure that describes the machine, not the code. One project
    per `dotnet test` also keeps the results of the others when one dies, and says which one it was.
    The projects are found, not listed, so a new test project cannot be left out.

    A zero exit code is not enough to count a project as passed: a run has been seen to end with exit 0 and no summary
    at all. Each project must also print a summary with at least one test. The summary line is localized, so the CLI
    language is pinned to English for the run.

.PARAMETER ExcludeHeavy
    Run the set CI runs (Slow, wall-clock-bound tests excluded).

.PARAMETER Coverage
    Collect code coverage.
#>

param(
    [switch]$ExcludeHeavy,
    [switch]$Coverage
)

Set-Location (Join-Path $PSScriptRoot "..")

$extra = @()
if ($ExcludeHeavy) { $extra += @("--filter", "Category!=LLM&Category!=Slow&Category!=E2E&Category!=Database") }
if ($Coverage) { $extra += @("--coverage", "--coverage-output-format", "cobertura") }
$env:DOTNET_CLI_UI_LANGUAGE = "en"

$failed = @()
foreach ($project in Get-ChildItem -Path tests -Filter *.Tests.csproj -Recurse -Depth 1 | Sort-Object FullName) {
    Write-Host "=== $($project.FullName)"
    $output = @()
    dotnet test $project.FullName --configuration Release --nologo @extra 2>&1 | Tee-Object -Variable output | Out-Host
    # [char]27 rather than the `e escape: Windows PowerShell 5.1 does not know `e.
    $esc = [char]27
    $plain = $output | ForEach-Object { "$_" -replace "$esc\[[0-9;]*m", "" }
    if ($LASTEXITCODE -ne 0) { $failed += $project.Name }
    elseif (-not ($plain -match '(?m)^\s*total: [1-9]\d*')) { $failed += "$($project.Name) (exit 0, but no test summary)" }
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "All test projects passed." -ForegroundColor Green
} else {
    Write-Host "Failed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
