# MLoop Job Management Scripts

Utilities for managing concurrent MLoop training jobs and preventing resource exhaustion.

## Overview

MLoop follows the Unix philosophy: each command runs as an independent process. For managing multiple training jobs, use these scripts to control resource usage.

## Scripts

### 1. `mloop-queue.sh` - Sequential Job Queue ⭐ Recommended

Execute multiple MLoop commands one after another.

**Usage**:
```bash
./mloop-queue.sh "COMMAND1" "COMMAND2" "COMMAND3"
```

**Example**:
```bash
./mloop-queue.sh \
  "train datasets/sales-2022.csv revenue --time 1800" \
  "train datasets/sales-2023.csv revenue --time 1800" \
  "evaluate models/staging/exp-001/model.zip datasets/test.csv revenue"
```

**Features**:
- ✅ Sequential execution (one at a time)
- ✅ Progress tracking with job numbers
- ✅ Error handling with continue/abort prompt
- ✅ Time tracking per job and total
- ✅ Color-coded output
- ✅ Summary report

**Output**:
```
╔════════════════════════════════════════╗
║   MLoop Sequential Job Queue           ║
╚════════════════════════════════════════╝

Total jobs: 3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1/3] Starting job
Command: mloop train datasets/sales-2022.csv revenue --time 1800
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Training output...]

✅ Job 1 complete (30m 15s)

[2/3] Starting job
...
```

---

### 2. `mloop-parallel.sh` - GNU Parallel Examples

Advanced parallel job execution using [GNU Parallel](https://www.gnu.org/software/parallel/).

**Prerequisites**:
```bash
# Ubuntu/Debian
sudo apt-get install parallel

# macOS
brew install parallel
```

**Usage**:
```bash
./mloop-parallel.sh [example-number]
```

**Available Examples**:

1. **Basic Sequential** - One job at a time (`-j 1`)
2. **Parallel Execution** - 2 jobs concurrently (`-j 2`) ⚠️ High CPU usage
3. **Resume on Failure** - Auto-resume from last successful job
4. **Different Time Budgets** - Each dataset gets different training time
5. **Progress Bar** - Visual progress tracking with ETA
6. **Dry Run** - Preview commands without execution

**Example 1: Sequential**:
```bash
./mloop-parallel.sh 1

# Executes:
# train dataset1.csv target --time 300
# train dataset2.csv target --time 600
# train dataset3.csv target --time 900
```

**Example 3: Resume on Failure**:
```bash
./mloop-parallel.sh 3

# Creates job log: /tmp/mloop-joblog.txt
# If a job fails, re-run to resume from failed job
```

**Example 5: Progress Bar**:
```bash
./mloop-parallel.sh 5

# Shows:
# Computers / CPU cores / Max jobs to run
# [===================>   ] 75%  ETA: 5m 30s
```

---

## Quick Reference

### Sequential Execution (Recommended)

```bash
# Simple queue
./mloop-queue.sh \
  "train data1.csv price --time 1800" \
  "train data2.csv price --time 1800"

# With evaluation
./mloop-queue.sh \
  "train datasets/train.csv price --time 1800" \
  "evaluate models/staging/exp-001/model.zip datasets/test.csv price" \
  "promote exp-001"
```

### Parallel Execution (Advanced)

```bash
# From file list
cat jobs.txt | parallel -j 1 mloop {}

# From datasets
parallel -j 1 mloop train {} price --time 1800 ::: dataset*.csv

# With job log (resume capability)
parallel --joblog jobs.log -j 1 mloop train {} ::: dataset*.csv
parallel --resume --joblog jobs.log -j 1 mloop train {} ::: dataset*.csv

# Progress bar
parallel --bar -j 1 mloop train {} ::: dataset*.csv

# Dry run (preview)
parallel --dry-run -j 1 mloop train {} ::: dataset*.csv
```

---

## Resource Management Tips

### 1. Estimate Resource Needs First

```bash
# Run one training job to measure
$ mloop train dataset1.csv price --time 1800 &
$ htop  # Monitor CPU/memory usage
```

### 2. Sequential is Safer

Default to sequential execution (`-j 1`) to prevent resource exhaustion.

### 3. Use Nice for Lower Priority

```bash
# Reduce CPU priority (0-19, higher = lower priority)
nice -n 19 mloop train dataset.csv price --time 1800
```

### 4. Set Memory Limits (Linux)

```bash
# Limit to 4GB memory
systemd-run --scope -p MemoryMax=4G mloop train dataset.csv price --time 1800
```

### 5. Pin to CPU Cores (Linux)

```bash
# Use cores 0-3 only
taskset -c 0-3 mloop train dataset.csv price --time 1800
```

---

## Common Patterns

### Pattern 1: Multiple Datasets, Same Settings

```bash
./mloop-queue.sh \
  "train datasets/jan-2024.csv revenue --time 1800" \
  "train datasets/feb-2024.csv revenue --time 1800" \
  "train datasets/mar-2024.csv revenue --time 1800"
```

### Pattern 2: Same Dataset, Different Time Budgets

```bash
./mloop-queue.sh \
  "train datasets/sales.csv revenue --time 300" \
  "train datasets/sales.csv revenue --time 600" \
  "train datasets/sales.csv revenue --time 1800"
```

### Pattern 3: Train → Evaluate → Promote

```bash
./mloop-queue.sh \
  "train datasets/train.csv price --time 1800" \
  "evaluate models/staging/exp-001/model.zip datasets/test.csv price" \
  "promote exp-001"
```

### Pattern 4: Multiple Experiments with Evaluation

```bash
./mloop-queue.sh \
  "train datasets/train.csv price --time 600 --metric r_squared" \
  "evaluate models/staging/exp-001/model.zip datasets/test.csv price" \
  "train datasets/train.csv price --time 1200 --metric rmse" \
  "evaluate models/staging/exp-002/model.zip datasets/test.csv price" \
  "train datasets/train.csv price --time 1800 --metric mae" \
  "evaluate models/staging/exp-003/model.zip datasets/test.csv price"
```

---

## Troubleshooting

### Script Permission Denied

```bash
chmod +x mloop-queue.sh mloop-parallel.sh
```

### GNU Parallel Not Found

```bash
# Ubuntu/Debian
sudo apt-get install parallel

# macOS
brew install parallel

# Other systems
# Download from: https://www.gnu.org/software/parallel/
```

### MLoop Command Not Found

```bash
# Install MLoop globally
dotnet tool install -g mloop

# Verify installation
mloop --version
```

### Jobs Running Slowly

- **Check CPU usage**: `htop` or `top`
- **Use sequential execution**: `-j 1` for GNU Parallel
- **Lower priority**: Use `nice -n 19`
- **Reduce time budget**: Shorter training times
- **Close other applications**: Free up resources

---

## Advanced GNU Parallel Tips

### Timeout Per Job

```bash
# Kill jobs taking longer than 1 hour
parallel --timeout 3600 -j 1 mloop train {} ::: dataset*.csv
```

### Different Parameters Per Dataset

```bash
# CSV format: dataset,label,time
# data1.csv,price,300
# data2.csv,revenue,600
# data3.csv,profit,900

parallel --colsep ',' -j 1 \
  mloop train {1} {2} --time {3} \
  :::: jobs.csv
```

### Email Notification

```bash
parallel -j 1 mloop train {} ::: dataset*.csv \
  && echo "Training complete" | mail -s "MLoop Jobs Done" you@example.com
```

### Retry Failed Jobs

```bash
# First run
parallel --joblog jobs.log -j 1 mloop train {} ::: dataset*.csv

# Retry only failed jobs
parallel --retry-failed --joblog jobs.log -j 1 mloop train {} ::: dataset*.csv
```

---

## See Also

- **[User Guide](../../docs/GUIDE.md)** - Complete MLoop documentation
- **[Concurrent Job Management](../../docs/GUIDE.md#concurrent-job-management)** - Detailed guide
- **[GNU Parallel Manual](https://www.gnu.org/software/parallel/man.html)** - Official documentation

---

**MLoop**: Clean Data In, Trained Model Out - That's It.
