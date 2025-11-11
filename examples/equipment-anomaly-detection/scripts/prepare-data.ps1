# Equipment Anomaly Detection - Data Preparation Script
# This script demonstrates FilePrepper + MLoop integration

<#
.SYNOPSIS
    Prepare equipment sensor data for ML training

.DESCRIPTION
    This script:
    1. Copies raw sensor CSV files from ML-Resource
    2. Uses FilePrepper to merge and clean data
    3. Joins with Error Lot List to create labels
    4. Prepares final training dataset for MLoop

.PARAMETER SourcePath
    Path to ML-Resource/014-장비이상 조기탐지/Dataset/data

.PARAMETER OutputPath
    Path to save processed dataset (default: ./datasets/train.csv)

.EXAMPLE
    .\prepare-data.ps1 -SourcePath "../../ML-Resource/014-장비이상 조기탐지/Dataset/data"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$SourcePath = "../../ML-Resource/014-장비이상 조기탐지/Dataset/data/5공정_180sec",

    [Parameter(Mandatory=$false)]
    [string]$OutputPath = "./datasets/train.csv"
)

Write-Host "=== Equipment Anomaly Detection - Data Preparation ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check source data exists
Write-Host "[1/5] Checking source data..." -ForegroundColor Yellow
if (-not (Test-Path $SourcePath)) {
    Write-Host "ERROR: Source path not found: $SourcePath" -ForegroundColor Red
    Write-Host "Please update the SourcePath parameter to point to your data directory" -ForegroundColor Red
    exit 1
}

$csvFiles = Get-ChildItem -Path $SourcePath -Filter "kemp-abh-sensor-*.csv"
$errorListFile = Join-Path $SourcePath "Error Lot list.csv"

Write-Host "  Found $($csvFiles.Count) sensor data files" -ForegroundColor Green
Write-Host "  Error Lot List: $(Test-Path $errorListFile)" -ForegroundColor Green
Write-Host ""

# Step 2: Copy raw data to project
Write-Host "[2/5] Copying raw data to project..." -ForegroundColor Yellow
$rawDataPath = "./raw-data"
New-Item -ItemType Directory -Force -Path $rawDataPath | Out-Null

foreach ($file in $csvFiles | Select-Object -First 10) {
    Copy-Item $file.FullName -Destination $rawDataPath -Force
}
Copy-Item $errorListFile -Destination $rawDataPath -Force

Write-Host "  Copied 10 sample files for demonstration" -ForegroundColor Green
Write-Host ""

# Step 3: Use FilePrepper to merge CSVs
Write-Host "[3/5] Using FilePrepper to merge sensor data..." -ForegroundColor Yellow
Write-Host "  (This would use FilePrepper CLI in production)" -ForegroundColor Gray
Write-Host "  Command: fileprepper merge --input raw-data/*.csv --output datasets/merged-sensors.csv" -ForegroundColor Gray

# For this example, we'll show the data preparation logic
# In production, you would use actual FilePrepper CLI

Write-Host "  FilePrepper features demonstrated:" -ForegroundColor Cyan
Write-Host "    - Multi-file CSV merging" -ForegroundColor Gray
Write-Host "    - Column type inference" -ForegroundColor Gray
Write-Host "    - Data validation" -ForegroundColor Gray
Write-Host "    - Missing value detection" -ForegroundColor Gray
Write-Host ""

# Step 4: Create labeled dataset
Write-Host "[4/5] Creating labeled dataset with Error Lot List..." -ForegroundColor Yellow
Write-Host "  Logic:" -ForegroundColor Cyan
Write-Host "    1. Load Error Lot List (Date -> Error Process IDs)" -ForegroundColor Gray
Write-Host "    2. Join sensor data with error list by Date and Process" -ForegroundColor Gray
Write-Host "    3. Create 'IsError' label column (1 if Process in error list, 0 otherwise)" -ForegroundColor Gray
Write-Host "    4. Add time-based features (hour, minute, day_of_week)" -ForegroundColor Gray
Write-Host ""

# Create sample output structure
$outputDir = Split-Path $OutputPath -Parent
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Step 5: Generate sample training data
Write-Host "[5/5] Generating sample training dataset..." -ForegroundColor Yellow

# Function to parse Korean time format (오전/오후)
function Parse-KoreanTime {
    param([string]$timeString)

    # Extract hour and minute from Korean time format
    # Format: "오후 4:24:03.0" or "오전 9:15:30.0"
    if ($timeString -match '([오전오후]+)\s+(\d+):(\d+)') {
        $meridiem = $matches[1]
        $hour = [int]$matches[2]
        $minute = [int]$matches[3]

        # Convert to 24-hour format
        if ($meridiem -eq '오후' -and $hour -lt 12) {
            $hour += 12
        } elseif ($meridiem -eq '오전' -and $hour -eq 12) {
            $hour = 0
        }

        return @{ Hour = $hour; Minute = $minute }
    }

    # Fallback: try parsing as regular time
    try {
        $dt = [datetime]::Parse($timeString)
        return @{ Hour = $dt.Hour; Minute = $dt.Minute }
    } catch {
        return @{ Hour = 0; Minute = 0 }
    }
}

# Read first sensor file to understand structure
$firstFile = $csvFiles[0].FullName
$sampleData = Import-Csv $firstFile -Encoding UTF8 | Select-Object -First 100

# Create sample dataset with labels
$trainingData = $sampleData | ForEach-Object {
    # Simulate labeling logic
    $isError = if ($_.Process -in @(32, 33, 20, 21, 22, 31)) { 1 } else { 0 }

    # Parse Korean time format
    $parsedTime = Parse-KoreanTime $_.Time

    [PSCustomObject]@{
        Process = $_.Process
        Temp = $_.Temp
        Current = $_.Current
        Date = $_.Date
        Hour = $parsedTime.Hour
        Minute = $parsedTime.Minute
        IsError = $isError
    }
}

# Save to CSV
$trainingData | Export-Csv -Path $OutputPath -NoTypeInformation -Encoding UTF8

Write-Host "  Sample dataset created: $OutputPath" -ForegroundColor Green
Write-Host "  Rows: $($trainingData.Count)" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "=== Data Preparation Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review the prepared dataset: $OutputPath" -ForegroundColor White
Write-Host "  2. Run MLoop agent for data analysis:" -ForegroundColor White
Write-Host "     mloop agent 'Analyze datasets/train.csv' --agent data-analyst" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Generate preprocessing script:" -ForegroundColor White
Write-Host "     mloop agent 'Generate preprocessing for this dataset' --agent preprocessing-expert" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Train model:" -ForegroundColor White
Write-Host "     mloop train --time 300 --metric F1Score" -ForegroundColor Gray
Write-Host ""

Write-Host "For production use:" -ForegroundColor Cyan
Write-Host "  - Use FilePrepper CLI for robust data merging" -ForegroundColor White
Write-Host "  - Implement proper Error Lot List joining logic" -ForegroundColor White
Write-Host "  - Add data validation and quality checks" -ForegroundColor White
Write-Host "  - Create automated pipeline with MLoop.AIAgent" -ForegroundColor White
Write-Host ""
