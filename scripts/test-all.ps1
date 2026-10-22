#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Runs every test project, one at a time.

.DESCRIPTION
    Serially, never two at once: several suites run real trainer searches on a ten-second budget, and
    under CPU load no trial completes — a failure that describes the machine, not the code. One project
    per `dotnet test` also keeps the results of the others when one dies, and says which one it was.
    The projects are found, not listed, so a new test project cannot be left out.

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
if ($Coverage) { $extra += @("--collect:XPlat Code Coverage") }

$failed = @()
foreach ($project in Get-ChildItem -Path tests -Filter *.Tests.csproj -Recurse -Depth 1 | Sort-Object FullName) {
    Write-Host "=== $($project.FullName)"
    dotnet test $project.FullName --configuration Release --nologo @extra
    if ($LASTEXITCODE -ne 0) { $failed += $project.Name }
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "All test projects passed." -ForegroundColor Green
} else {
    Write-Host "Failed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
