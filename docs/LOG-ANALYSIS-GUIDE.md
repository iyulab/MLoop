# Log Analysis with MLoop

A practical guide for building ML models from log data using MLoop.

## CSV Format Requirements

### RFC 4180 Quoting Rules

MLoop uses standard CSV format. Fields containing commas, double quotes, or newlines **must** be wrapped in double quotes:

```csv
Content,Label
"Feb 20 14:20:11 web-12 systemd[1]: cpagent.service: Failed with result 'exit-code'.",error
"Feb 20 14:20:15 web-12 systemd[1]: Started Session 42 of user admin.",normal
```

If a field contains double quotes, escape them as `""`:

```csv
Content,Label
"He said ""hello"" to me",normal
```

**Common mistake**: Unquoted fields with commas cause column count mismatch errors.

## Recommended Column Structure

### Keep it simple: Content + Label

For log classification, the minimal and recommended structure is:

```csv
Content,Label
"<log message 1>",<category>
"<log message 2>",<category>
```

### Avoid unnecessary columns

| Column | Include? | Reason |
|--------|----------|--------|
| Content (log text) | Yes | Primary feature for classification |
| Label | Yes (training only) | Target variable |
| Timestamp | No | Auto-excluded if detected as DateTime; string timestamps add noise |
| Hostname | No | Usually irrelevant to log classification |
| PID | No | Process IDs are not predictive |
| Severity/Level | Optional | Useful if available as structured data |

**Key rule**: Every feature column used during training must also be present during prediction. Only include columns that are genuinely predictive.

### Prediction CSV

Prediction data should have the same feature columns as training, but **without** the Label column:

```csv
Content
"Feb 20 15:30:00 app-01 kernel: Out of memory: Kill process 1234"
"Feb 20 15:30:05 app-01 sshd[5678]: Accepted publickey for admin"
```

## Minimum Data Requirements

| Task Type | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Binary Classification | 10 per class (20 total) | 50+ per class | Extreme imbalance (>50:1) may cause AUC errors |
| Multiclass Classification | 5 per class | 15+ per class | Fewer than 5 causes cross-validation failures |
| Regression | 30 rows | 100+ rows | Also need 10x feature count |

MLoop validates data quality before training and warns about insufficient data.

## Step-by-Step Workflow

### 1. Prepare training data

Create `train.csv` with labeled log examples:

```csv
Content,Label
"systemd[1]: cpagent.service: Failed with result 'exit-code'.",error
"sshd[5678]: Accepted publickey for admin from 10.0.0.1",normal
"kernel: Out of memory: Kill process 1234 (java)",error
"systemd[1]: Started Session 42 of user admin.",normal
```

Aim for balanced classes and sufficient samples per class.

### 2. Initialize project

```bash
mloop init my-log-project --task binary-classification
```

Copy `train.csv` to `my-log-project/datasets/`.

### 3. Train

```bash
cd my-log-project
mloop train datasets/train.csv --label Label --task binary-classification
```

### 4. Predict

```bash
mloop predict datasets/predict.csv
```

By default, prediction output includes both original features and prediction results (`--include-features` is on by default). Use `--include-features false` for raw prediction output only.

### 5. Review results

Output in `predictions/` includes:

| Column | Description |
|--------|-------------|
| Content | Original log message (when --include-features is true) |
| PredictedLabel | Model's prediction |
| Score | Confidence score |
| Probability | Probability estimate (classification only) |

## Handling Categorical vs Text Data

MLoop automatically detects column types:

- **Categorical**: Low-cardinality values (e.g., severity levels: info, warn, error)
- **Text**: High-cardinality free-text (e.g., log messages)

Log message content is classified as **Text** when unique values exceed 50% of total rows. With very small training sets, it may be misclassified as Categorical, causing prediction failures. Solution: use more training data (50+ rows recommended).

If you encounter categorical mismatch errors during prediction:

```bash
# Auto-handle unknown categorical values
mloop predict data.csv --unknown-strategy auto

# Replace unknown values with empty string
mloop predict data.csv --unknown-strategy use-missing
```

## Tips

1. **Quote all log content** in CSV files to avoid parsing issues
2. **Exclude timestamps** from training features unless time-of-day patterns matter
3. **Use sufficient training data** - at least 50 rows per class for reliable results
4. **Check data quality** with `mloop info datasets/train.csv --analyze` before training
5. **Use `--include-features`** (default) to trace predictions back to source logs
