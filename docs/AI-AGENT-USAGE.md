# AI Agent Usage Guide

Complete guide for using MLoop's AI-powered agents for interactive ML workflow assistance.

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Agent Overview](#2-agent-overview)
3. [Configuration](#3-configuration)
4. [Usage Examples](#4-usage-examples)
5. [Advanced Usage](#5-advanced-usage)
6. [Workflows](#6-workflows)
7. [Troubleshooting](#7-troubleshooting)
8. [Best Practices](#8-best-practices)

---

## 1. Quick Start

### 1.1 Prerequisites

- MLoop CLI installed (`dotnet tool install -g mloop`)
- At least one LLM provider configured

### 1.2 First Steps

```bash
# Step 1: Configure LLM provider (choose one)
export ANTHROPIC_API_KEY=sk-ant-your-key
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Step 2: Chat with an agent
mloop agent chat data-analyst "What can you help me with?"

# Step 3: Analyze your data
mloop agent chat data-analyst "Analyze datasets/train.csv"
```

### 1.3 Available Commands

| Command | Description |
|---------|-------------|
| `mloop agent chat <agent> <query>` | Single query to agent |
| `mloop agent stream <agent> <query>` | Streaming response |
| `mloop agent list` | List available agents |

---

## 2. Agent Overview

### 2.1 DataAnalystAgent

**Purpose**: Dataset analysis, statistics, ML readiness assessment

**Best For**:
- Understanding new datasets
- Identifying data quality issues
- Getting preprocessing recommendations
- ML readiness assessment

**Example Queries**:
```bash
mloop agent chat data-analyst "Analyze datasets/customer-data.csv"
mloop agent chat data-analyst "What's the distribution of the target column?"
mloop agent chat data-analyst "Are there any missing values or outliers?"
mloop agent chat data-analyst "Is this dataset ready for ML training?"
```

**Capabilities**:
- Dataset structure analysis (rows, columns, types)
- Statistical summaries (mean, std, quartiles)
- Missing value detection and patterns
- Outlier identification
- Class imbalance detection
- Feature correlation analysis
- ML readiness scoring

### 2.2 PreprocessingExpertAgent

**Purpose**: Generate C# preprocessing scripts for MLoop

**Best For**:
- Creating data transformation scripts
- Handling complex preprocessing needs
- Multi-file operations (join, merge)
- Feature engineering

**Example Queries**:
```bash
mloop agent chat preprocessing-expert "Handle missing values in the Age column"
mloop agent chat preprocessing-expert "Join customer.csv and orders.csv on customer_id"
mloop agent chat preprocessing-expert "Create feature: days_since_last_purchase"
mloop agent chat preprocessing-expert "Convert wide format to long format"
```

**Capabilities**:
- Generate `IPreprocessingScript` implementations
- Sequential naming (01_*, 02_*, 03_*)
- Multi-file join/merge operations
- Wide-to-long transformations
- Feature engineering scripts
- Data cleaning transformations

**Output Format**:
```csharp
// Generated: .mloop/scripts/preprocess/01_handle_missing.cs
using MLoop.Extensibility;

public class HandleMissingValues : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        // Implementation here
    }
}
```

### 2.3 ModelArchitectAgent

**Purpose**: ML problem classification and AutoML configuration

**Best For**:
- Determining problem type
- Selecting optimal metrics
- Configuring training parameters
- Understanding ML.NET trainers

**Example Queries**:
```bash
mloop agent chat model-architect "I have customer data and want to predict churn"
mloop agent chat model-architect "What metric should I use for imbalanced classification?"
mloop agent chat model-architect "Recommend training time for 100K rows"
mloop agent chat model-architect "Which trainers work best for regression?"
```

**Capabilities**:
- Problem type classification
- Metric selection (Accuracy, F1, AUC, RMSE, etc.)
- Time budget recommendations
- Trainer selection guidance
- Feature engineering suggestions
- `mloop train` command generation

### 2.4 MLOpsManagerAgent

**Purpose**: End-to-end workflow orchestration

**Best For**:
- Complete ML pipelines
- Project initialization
- Training execution
- Model deployment
- Experiment management

**Example Queries**:
```bash
mloop agent chat mlops-manager "Initialize a new project for customer churn prediction"
mloop agent chat mlops-manager "Train a model on datasets/train.csv with target 'Churned'"
mloop agent chat mlops-manager "List all experiments and their metrics"
mloop agent chat mlops-manager "Promote experiment exp-003 to production"
```

**Capabilities**:
- Project initialization (`mloop init`)
- Model training (`mloop train`)
- Model evaluation (`mloop evaluate`)
- Batch prediction (`mloop predict`)
- Experiment listing and comparison
- Model promotion to production
- Workflow orchestration

---

## 3. Configuration

### 3.1 Environment Variables

Create a `.env` file in your project root:

```bash
# Option 1: GPUStack (Local - 89% cost savings)
GPUSTACK_ENDPOINT=http://localhost:8080/v1
GPUSTACK_API_KEY=your-gpustack-key
GPUSTACK_MODEL=llama-3.1-8b

# Option 2: Anthropic (Recommended for production)
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Option 3: Azure OpenAI (Enterprise)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_MODEL=gpt-4o

# Option 4: OpenAI (Development)
OPENAI_API_KEY=sk-proj-your-key
OPENAI_MODEL=gpt-4o-mini
```

### 3.2 Provider Priority

MLoop checks providers in this order:
1. **GPUStack** - Local deployment, lowest cost
2. **Anthropic** - Best quality, recommended for production
3. **Azure OpenAI** - Enterprise compliance
4. **OpenAI** - General development

### 3.3 Cost Comparison

| Provider | Input (per 1M) | Output (per 1M) | Monthly Estimate* |
|----------|----------------|-----------------|-------------------|
| GPUStack (Llama 3.1 8B) | ~$1 | ~$1 | $20-50 |
| Anthropic Claude 3.5 | $3 | $15 | $300-600 |
| OpenAI GPT-4o-mini | $0.15 | $0.60 | $50-100 |
| Azure OpenAI GPT-4o | $10 | $30 | $800-1500 |

*Based on 10-20M tokens/month (medium usage)

### 3.4 GPUStack Setup (Local)

```bash
# Docker (requires GPU)
docker run -d -p 8080:8080 --gpus all gpustack/gpustack:latest

# Or pip install
pip install gpustack
gpustack start --port 8080

# Verify
curl http://localhost:8080/health
```

---

## 4. Usage Examples

### 4.1 Data Analysis Workflow

```bash
# Step 1: Quick overview
mloop agent chat data-analyst "Give me an overview of datasets/customers.csv"

# Response includes:
# - Row/column count
# - Column types
# - Missing value summary
# - Basic statistics

# Step 2: Detailed analysis
mloop agent chat data-analyst "Analyze the target column 'Churned' for class imbalance"

# Step 3: ML readiness check
mloop agent chat data-analyst "Is this dataset ready for ML? What preprocessing is needed?"
```

### 4.2 Preprocessing Workflow

```bash
# Step 1: Identify needs
mloop agent chat data-analyst "What preprocessing does datasets/raw.csv need?"

# Step 2: Generate scripts
mloop agent chat preprocessing-expert "Handle missing values: fill numeric with median, categorical with mode"

# Step 3: Save script
# Copy the generated code to .mloop/scripts/preprocess/01_handle_missing.cs

# Step 4: Generate next script
mloop agent chat preprocessing-expert "Encode categorical columns: Gender, Country, ProductType"

# Step 5: Run preprocessing
mloop preprocess --input datasets/raw.csv --output datasets/train.csv
```

### 4.3 Training Workflow

```bash
# Step 1: Get recommendations
mloop agent chat model-architect "Binary classification with 50K rows, imbalanced classes"

# Step 2: Initialize project
mloop agent chat mlops-manager "Initialize project 'churn-prediction' for binary classification"

# Step 3: Train model
mloop agent chat mlops-manager "Train model on datasets/train.csv, target 'Churned', 5 minutes"

# Step 4: Evaluate
mloop agent chat mlops-manager "Evaluate the latest experiment"

# Step 5: Promote if good
mloop agent chat mlops-manager "Promote exp-001 to production"
```

### 4.4 Complete E2E Workflow

```bash
# Analysis phase
mloop agent chat data-analyst "Analyze datasets/sales.csv for predicting 'Revenue'"

# Preprocessing phase
mloop agent chat preprocessing-expert "Generate all preprocessing scripts based on the analysis"

# Configuration phase
mloop agent chat model-architect "Recommend configuration for this regression problem"

# Execution phase
mloop agent chat mlops-manager "Execute complete training workflow:
- Initialize project 'sales-forecast'
- Run preprocessing
- Train model for 10 minutes
- Evaluate and report results"
```

---

## 5. Advanced Usage

### 5.1 Streaming Responses

For long responses or real-time feedback:

```bash
# Stream analysis results
mloop agent stream data-analyst "Detailed statistical analysis of datasets/large-dataset.csv"

# Stream preprocessing script generation
mloop agent stream preprocessing-expert "Generate comprehensive preprocessing pipeline"
```

### 5.2 Complex Queries

```bash
# Multi-step analysis
mloop agent chat data-analyst "
1. Analyze datasets/customers.csv
2. Identify top 5 most important features for predicting 'Churned'
3. Check for multicollinearity
4. Recommend feature engineering
"

# Conditional workflow
mloop agent chat mlops-manager "
If datasets/train.csv exists, train model on it.
Otherwise, preprocess datasets/raw.csv first.
Use binary classification with F1 score optimization.
Time limit: 5 minutes.
"
```

### 5.3 Context Building

```bash
# Provide context in query
mloop agent chat model-architect "
Dataset: 100K rows, 50 features, 5% positive class
Problem: Predict customer churn
Constraints:
- Must optimize for recall (minimize false negatives)
- Model must be explainable
- Training time limit: 10 minutes
Recommend the best configuration.
"
```

### 5.4 Batch Operations

```bash
# Multiple datasets
for dataset in datasets/*.csv; do
    mloop agent chat data-analyst "Quick analysis of $dataset" >> analysis_report.md
done

# Multiple experiments
for time in 60 120 300; do
    mloop agent chat mlops-manager "Train with time=$time seconds, name=exp-${time}s"
done
```

---

## 6. Workflows

### 6.1 New Project Setup

```bash
# 1. Create project directory
mkdir my-ml-project && cd my-ml-project

# 2. Copy your data
cp /path/to/data.csv datasets/

# 3. Analyze data
mloop agent chat data-analyst "Analyze datasets/data.csv and recommend ML approach"

# 4. Generate preprocessing (if needed)
mloop agent chat preprocessing-expert "Create preprocessing scripts based on analysis"

# 5. Initialize MLoop project
mloop init my-project --task binary-classification --label target

# 6. Train
mloop train datasets/data.csv --label target --time 300
```

### 6.2 Model Improvement

```bash
# 1. Analyze current results
mloop agent chat mlops-manager "List all experiments with metrics"

# 2. Get improvement suggestions
mloop agent chat model-architect "Current best F1 is 0.75. How can I improve?"

# 3. Try feature engineering
mloop agent chat preprocessing-expert "Create interaction features for top 5 predictors"

# 4. Retrain with new features
mloop agent chat mlops-manager "Train new model with enhanced features, 10 minutes"

# 5. Compare results
mloop agent chat mlops-manager "Compare exp-001 vs exp-002"
```

### 6.3 Production Deployment

```bash
# 1. Evaluate production candidate
mloop agent chat mlops-manager "Detailed evaluation of exp-003"

# 2. Compare with current production
mloop agent chat mlops-manager "Compare exp-003 with current production model"

# 3. Promote if better
mloop agent chat mlops-manager "Promote exp-003 to production"

# 4. Verify deployment
mloop agent chat mlops-manager "Verify production model is updated"
```

---

## 7. Troubleshooting

### 7.1 Common Errors

**Error: "No LLM provider credentials found"**
```bash
# Solution: Set environment variables
export ANTHROPIC_API_KEY=sk-ant-your-key
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Or create .env file
echo 'ANTHROPIC_API_KEY=sk-ant-your-key' >> .env
echo 'ANTHROPIC_MODEL=claude-3-5-sonnet-20241022' >> .env
```

**Error: "Connection refused to GPUStack"**
```bash
# Solution: Verify GPUStack is running
curl http://localhost:8080/health

# Start GPUStack
docker run -d -p 8080:8080 --gpus all gpustack/gpustack:latest
```

**Error: "Rate limit exceeded"**
```bash
# Solution: Switch to GPUStack (unlimited) or wait
# Or upgrade API plan with provider
```

**Error: "Unknown agent: xyz"**
```bash
# Solution: Check available agents
mloop agent list

# Valid agents: data-analyst, preprocessing-expert, model-architect, mlops-manager
```

**Error: "Dataset not found"**
```bash
# Solution: Verify file path
ls -la datasets/

# Use absolute path if needed
mloop agent chat data-analyst "Analyze /full/path/to/data.csv"
```

### 7.2 Debugging

```bash
# Verbose output
mloop agent chat data-analyst "test" --verbose

# Check provider detection
env | grep -E "(ANTHROPIC|OPENAI|AZURE|GPUSTACK)"

# Test connectivity
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages \
     -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

### 7.3 Performance Issues

**Slow responses**:
- Use GPUStack for local deployment
- Try smaller models (gpt-4o-mini vs gpt-4o)
- Use streaming for long responses

**High costs**:
- Switch to GPUStack (local)
- Use OpenAI gpt-4o-mini
- Reduce query complexity

---

## 8. Best Practices

### 8.1 Query Optimization

**DO**:
```bash
# Be specific
mloop agent chat data-analyst "Analyze missing values in columns: Age, Income, Score"

# Provide context
mloop agent chat model-architect "Binary classification, 10K rows, 3% positive class"

# Use structured queries
mloop agent chat preprocessing-expert "
Task: Handle missing values
Columns: Age (numeric), City (categorical)
Strategy: median for numeric, mode for categorical
"
```

**DON'T**:
```bash
# Too vague
mloop agent chat data-analyst "analyze data"

# Missing context
mloop agent chat model-architect "what's a good metric?"

# Too complex in one query
mloop agent chat mlops-manager "do everything"
```

### 8.2 Workflow Organization

**Recommended Pattern**:
1. **Analyze** → Use data-analyst for understanding
2. **Plan** → Use model-architect for strategy
3. **Prepare** → Use preprocessing-expert for data prep
4. **Execute** → Use mlops-manager for operations
5. **Iterate** → Repeat with improvements

### 8.3 Cost Management

```bash
# Use streaming for exploration (same cost, better UX)
mloop agent stream data-analyst "detailed analysis..."

# Use specific queries (fewer tokens)
mloop agent chat data-analyst "missing value count per column"

# Cache results locally
mloop agent chat data-analyst "analyze data.csv" > analysis.md
```

### 8.4 Security

```bash
# Use .env file (not exported vars)
echo 'ANTHROPIC_API_KEY=your-key' >> .env

# Add .env to .gitignore
echo '.env' >> .gitignore

# Use GPUStack for sensitive data
# Data stays local, never sent to cloud
```

---

## Appendix A: Command Reference

### Agent Commands

```bash
# Chat (single query)
mloop agent chat <agent-name> "<query>"

# Stream (real-time response)
mloop agent stream <agent-name> "<query>"

# List agents
mloop agent list

# Help
mloop agent --help
```

### Agent Names

| Name | Alias | Description |
|------|-------|-------------|
| `data-analyst` | `da` | Dataset analysis |
| `preprocessing-expert` | `pe` | Script generation |
| `model-architect` | `ma` | ML configuration |
| `mlops-manager` | `mm` | Workflow orchestration |

### Examples

```bash
# Full name
mloop agent chat data-analyst "analyze data"

# With streaming
mloop agent stream preprocessing-expert "generate script"

# Complex query
mloop agent chat mlops-manager "train model on train.csv with target=Label time=300"
```

---

## Appendix B: Model Recommendations

### By Dataset Size

| Rows | Recommended Time | Metric Focus |
|------|------------------|--------------|
| < 1K | 30-60s | Accuracy |
| 1K-10K | 60-120s | F1/AUC |
| 10K-100K | 120-300s | AUC |
| > 100K | 300-600s | AUC |

### By Problem Type

| Problem | Recommended Metric | Key Trainer |
|---------|-------------------|-------------|
| Binary Classification | F1/AUC | LightGbm |
| Multiclass Classification | MacroF1 | LightGbm |
| Regression | RMSE/R² | FastTree |
| Imbalanced Classification | F1 | LightGbm + SMOTE |

### By Business Goal

| Goal | Metric | Rationale |
|------|--------|-----------|
| Minimize false negatives | Recall | Don't miss positive cases |
| Minimize false positives | Precision | Avoid wrong positives |
| Balance both | F1 | Harmonic mean |
| Probability calibration | AUC | Ranking quality |

---

**Version**: 1.0.0
**Last Updated**: 2024-12-08
**Related**: [AI-AGENT-ARCHITECTURE.md](AI-AGENT-ARCHITECTURE.md), [AI-AGENTS.md](AI-AGENTS.md)
