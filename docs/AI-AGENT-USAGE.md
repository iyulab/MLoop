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

# Step 2: Query with specific agent
mloop agent "What can you help me with?" --agent data-analyst

# Step 3: Analyze your data
mloop agent "Analyze datasets/train.csv" --agent data-analyst

# Step 4: Interactive conversation mode
mloop agent --interactive
```

### 1.3 Available Commands

| Command | Description |
|---------|-------------|
| `mloop agent "query" --agent <name>` | Single query to specific agent |
| `mloop agent "query"` | Auto-select agent based on query |
| `mloop agent --interactive` | Interactive conversation mode |
| `mloop agent --list-agents` | List all available agents |
| `mloop agent workflow run <file>` | Execute workflow definition |
| `mloop agent workflow list` | List available workflows |

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
mloop agent "Analyze datasets/customer-data.csv" --agent data-analyst
mloop agent "What's the distribution of the target column?" -a data-analyst
mloop agent "Are there any missing values or outliers?" -a data-analyst
mloop agent "Is this dataset ready for ML training?" -a data-analyst
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

**Purpose**: Generate C# preprocessing scripts and orchestrate incremental preprocessing workflows

**Best For**:
- Creating data transformation scripts
- Incremental preprocessing for large datasets (100K+ records)
- Handling complex preprocessing needs
- Multi-file operations (join, merge)
- Feature engineering

**Example Queries**:
```bash
# Standard preprocessing
mloop agent "Handle missing values in the Age column" --agent preprocessing-expert
mloop agent "Join customer.csv and orders.csv on customer_id" -a preprocessing-expert
mloop agent "Create feature: days_since_last_purchase" -a preprocessing-expert

# Incremental preprocessing (large datasets)
mloop agent "Preprocess datasets/large-logs.csv incrementally" -a preprocessing-expert
mloop preprocess data.csv --incremental --confidence-threshold 0.98
```

**Capabilities**:
- **Script Generation**: Generate `IPreprocessingScript` implementations with sequential naming (01_*, 02_*, 03_*)
- **Incremental Preprocessing**: 5-stage progressive sampling workflow for large datasets
  - Stage 1 (0.1%): Initial exploration and schema discovery
  - Stage 2 (0.5%): Pattern validation and auto-fix application
  - Stage 3 (1.5%): Human-in-the-loop decisions for business logic
  - Stage 4 (2.5%): Confidence checkpoint and final approval
  - Stage 5 (100%): Bulk processing with validated rules
- **Rule Discovery**: Automatic detection of 8 pattern types (missing values, outliers, encoding issues, etc.)
- **Confidence Scoring**: Statistical validation with 98%+ confidence before bulk application
- **Multi-file Operations**: Join, merge, and transformation scripts
- **Feature Engineering**: Custom feature creation scripts

**Incremental Preprocessing Workflow**:
```bash
# Start incremental preprocessing
mloop agent "preprocess manufacturing-logs.csv incrementally" --agent preprocessing-expert

# The agent will:
# 1. Analyze 0.1% sample (100 records) → discover patterns
# 2. Validate with 0.5% sample → apply auto-fixes
# 3. Request human decisions for business logic
# 4. Confirm 98%+ confidence at 2.5% sample
# 5. Apply validated rules to all 100% records

# Options:
# --confidence-threshold 0.98    # Minimum confidence for auto-apply
# --max-error-rate 0.01          # Stop if error rate exceeds 1%
# --skip-hitl                     # Use defaults (no prompts)
```

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

**Deliverables** (Incremental Mode):
- `cleaned_data.csv` - Preprocessed dataset
- `01_preprocess_*.cs` - Reusable preprocessing script
- `exceptions.json` - Records that didn't fit rules
- `preprocessing_report.md` - Detailed analysis report

### 2.3 ModelArchitectAgent

**Purpose**: ML problem classification and AutoML configuration

**Best For**:
- Determining problem type
- Selecting optimal metrics
- Configuring training parameters
- Understanding ML.NET trainers

**Example Queries**:
```bash
mloop agent "I have customer data and want to predict churn" --agent model-architect
mloop agent "What metric should I use for imbalanced classification?" -a model-architect
mloop agent "Recommend training time for 100K rows" -a model-architect
mloop agent "Which trainers work best for regression?" -a model-architect
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
mloop agent "Initialize a new project for customer churn prediction" --agent mlops-manager
mloop agent "Train a model on datasets/train.csv with target 'Churned'" -a mlops-manager
mloop agent "List all experiments and their metrics" -a mlops-manager
mloop agent "Promote experiment exp-003 to production" -a mlops-manager
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
mloop agent "Give me an overview of datasets/customers.csv" --agent data-analyst

# Response includes:
# - Row/column count
# - Column types
# - Missing value summary
# - Basic statistics

# Step 2: Detailed analysis
mloop agent "Analyze the target column 'Churned' for class imbalance" -a data-analyst

# Step 3: ML readiness check
mloop agent "Is this dataset ready for ML? What preprocessing is needed?" -a data-analyst
```

### 4.2 Preprocessing Workflow

```bash
# Step 1: Identify needs
mloop agent "What preprocessing does datasets/raw.csv need?" --agent data-analyst

# Step 2: Generate scripts
mloop agent "Handle missing values: fill numeric with median, categorical with mode" -a preprocessing-expert

# Step 3: Save script
# Copy the generated code to .mloop/scripts/preprocess/01_handle_missing.cs

# Step 4: Generate next script
mloop agent "Encode categorical columns: Gender, Country, ProductType" -a preprocessing-expert

# Step 5: Run preprocessing
mloop preprocess --input datasets/raw.csv --output datasets/train.csv
```

### 4.3 Training Workflow

```bash
# Step 1: Get recommendations
mloop agent "Binary classification with 50K rows, imbalanced classes" --agent model-architect

# Step 2: Initialize project
mloop agent "Initialize project 'churn-prediction' for binary classification" -a mlops-manager

# Step 3: Train model
mloop agent "Train model on datasets/train.csv, target 'Churned', 5 minutes" -a mlops-manager

# Step 4: Evaluate
mloop agent "Evaluate the latest experiment" -a mlops-manager

# Step 5: Promote if good
mloop agent "Promote exp-001 to production" -a mlops-manager
```

### 4.4 Complete E2E Workflow

```bash
# Analysis phase
mloop agent "Analyze datasets/sales.csv for predicting 'Revenue'" --agent data-analyst

# Preprocessing phase
mloop agent "Generate all preprocessing scripts based on the analysis" -a preprocessing-expert

# Configuration phase
mloop agent "Recommend configuration for this regression problem" -a model-architect

# Execution phase (use interactive mode for complex multi-step tasks)
mloop agent --interactive --agent mlops-manager
# Then: Execute complete training workflow:
# - Initialize project 'sales-forecast'
# - Run preprocessing
# - Train model for 10 minutes
# - Evaluate and report results
```

---

## 5. Advanced Usage

### 5.1 Response Streaming

Responses are streamed in real-time by default. This provides better UX for long analyses:

```bash
# Responses stream automatically as they're generated
mloop agent "Detailed statistical analysis of datasets/large-dataset.csv" --agent data-analyst

# Long preprocessing script generation
mloop agent "Generate comprehensive preprocessing pipeline" -a preprocessing-expert
```

### 5.2 Complex Queries

```bash
# Multi-step analysis
mloop agent "
1. Analyze datasets/customers.csv
2. Identify top 5 most important features for predicting 'Churned'
3. Check for multicollinearity
4. Recommend feature engineering
" --agent data-analyst

# Conditional workflow
mloop agent "
If datasets/train.csv exists, train model on it.
Otherwise, preprocess datasets/raw.csv first.
Use binary classification with F1 score optimization.
Time limit: 5 minutes.
" -a mlops-manager
```

### 5.3 Context Building

```bash
# Provide context in query
mloop agent "
Dataset: 100K rows, 50 features, 5% positive class
Problem: Predict customer churn
Constraints:
- Must optimize for recall (minimize false negatives)
- Model must be explainable
- Training time limit: 10 minutes
Recommend the best configuration.
" --agent model-architect
```

### 5.4 Batch Operations

```bash
# Multiple datasets
for dataset in datasets/*.csv; do
    mloop agent "Quick analysis of $dataset" --agent data-analyst >> analysis_report.md
done

# Multiple experiments
for time in 60 120 300; do
    mloop agent "Train with time=$time seconds, name=exp-${time}s" -a mlops-manager
done
```

### 5.5 Content Guardrails (v1.2.0+)

MLoop integrates Ironbees v0.2.0 Guardrails for content validation and security.

#### Enabling Guardrails (Programmatic)

```csharp
// Create orchestrator
var orchestrator = IronbeesOrchestrator.CreateFromEnvironment();
await orchestrator.InitializeAsync();

// Enable standard ML guardrails (PII, injection, length)
orchestrator.UseStandardGuardrails();

// Or enable strict guardrails for production
orchestrator.UseStrictGuardrails();

// Or create custom pipeline
orchestrator.UseGuardrails(MLGuardrails.CreateStandardPipeline(
    enablePII: true,
    enableInjection: true,
    enableLength: true,
    maxInputLength: 50000));
```

#### Built-in Guardrail Types

| Guardrail | Purpose | Default |
|-----------|---------|---------|
| **PII Detector** | Email, SSN, credit card, phone detection | Input + Output |
| **Injection Detector** | SQL, command, path traversal prevention | Input only |
| **Length Validator** | DoS prevention via length limits | Input only |
| **ML Keyword Filter** | Blocks sensitive ML operations | Input only |

#### Custom Guardrails

```csharp
// Create individual guardrails
var piiGuardrail = MLGuardrails.CreatePIIGuardrail();
var injectionGuardrail = MLGuardrails.CreateInjectionGuardrail();
var lengthGuardrail = MLGuardrails.CreateLengthGuardrail(maxLength: 10000);
var keywordGuardrail = MLGuardrails.CreateMLKeywordGuardrail(
    new[] { "custom-blocked-term" });

// Manual validation
var result = await orchestrator.ValidateInputAsync("user query");
if (!result?.IsAllowed ?? true)
{
    // Handle violation
    Console.WriteLine($"Blocked: {result.AllViolations.First().Description}");
}
```

#### Guardrail Response Handling

When guardrails block content:
- **Input Blocked**: Returns `[Guardrail Violation] Your request was blocked: {reason}`
- **Output Filtered**: Returns `[Content Filtered] The response was filtered due to policy violations.`

---

## 6. Workflows

### 6.1 New Project Setup

```bash
# 1. Create project directory
mkdir my-ml-project && cd my-ml-project

# 2. Copy your data
cp /path/to/data.csv datasets/

# 3. Analyze data
mloop agent "Analyze datasets/data.csv and recommend ML approach" --agent data-analyst

# 4. Generate preprocessing (if needed)
mloop agent "Create preprocessing scripts based on analysis" -a preprocessing-expert

# 5. Initialize MLoop project
mloop init my-project --task binary-classification --label target

# 6. Train
mloop train datasets/data.csv --label target --time 300
```

### 6.2 Model Improvement

```bash
# 1. Analyze current results
mloop agent "List all experiments with metrics" --agent mlops-manager

# 2. Get improvement suggestions
mloop agent "Current best F1 is 0.75. How can I improve?" -a model-architect

# 3. Try feature engineering
mloop agent "Create interaction features for top 5 predictors" -a preprocessing-expert

# 4. Retrain with new features
mloop agent "Train new model with enhanced features, 10 minutes" -a mlops-manager

# 5. Compare results
mloop agent "Compare exp-001 vs exp-002" -a mlops-manager
```

### 6.3 Production Deployment

```bash
# 1. Evaluate production candidate
mloop agent "Detailed evaluation of exp-003" --agent mlops-manager

# 2. Compare with current production
mloop agent "Compare exp-003 with current production model" -a mlops-manager

# 3. Promote if better
mloop agent "Promote exp-003 to production" -a mlops-manager

# 4. Verify deployment
mloop agent "Verify production model is updated" -a mlops-manager
```

### 6.4 Automated Workflows (YAML)

MLoop supports automated multi-agent workflows defined in YAML files.

**Create a workflow file** (`.mloop/workflows/ml-pipeline.yaml`):

```yaml
name: MLPipeline
version: "1.0"
description: End-to-end ML workflow

agents:
  - ref: data-analyst
    alias: analyst
  - ref: preprocessing-expert
    alias: prepper
  - ref: model-architect
    alias: architect

states:
  - id: START
    type: Start
    next: ANALYZE

  - id: ANALYZE
    type: Agent
    executor: analyst
    next: PREPROCESS

  - id: PREPROCESS
    type: Agent
    executor: prepper
    next: CONFIGURE

  - id: CONFIGURE
    type: Agent
    executor: architect
    next: REVIEW

  - id: REVIEW
    type: HumanGate
    approvalMessage: "Review configuration before training?"
    next: END

  - id: END
    type: Terminal

settings:
  defaultTimeout: PT10M
  enableCheckpointing: true
```

**Execute the workflow**:

```bash
# List available workflows
mloop agent workflow list

# Validate workflow syntax before running
mloop agent workflow validate .mloop/workflows/ml-pipeline.yaml

# Run the workflow
mloop agent workflow run .mloop/workflows/ml-pipeline.yaml --input "Analyze customer-data.csv"

# Check execution status
mloop agent workflow status
```

**Workflow features**:
- **Agent States**: Execute specific agents with defined roles
- **Human Gates**: Pause for manual approval before continuing
- **Checkpointing**: Resume workflows after interruption
- **State Transitions**: Define execution flow between states

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
# Solution: List available agents
mloop agent --list-agents

# Or check in interactive mode
mloop agent --interactive
# Then type: /agents

# Valid agents: data-analyst, preprocessing-expert, model-architect, mlops-manager
```

**Error: "Dataset not found"**
```bash
# Solution: Verify file path
ls -la datasets/

# Use absolute path if needed
mloop agent "Analyze /full/path/to/data.csv" --agent data-analyst
```

### 7.2 Debugging

```bash
# Test with simple query
mloop agent "test" --agent data-analyst

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
- Responses are streamed by default for better UX

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
mloop agent "Analyze missing values in columns: Age, Income, Score" --agent data-analyst

# Provide context
mloop agent "Binary classification, 10K rows, 3% positive class" -a model-architect

# Use structured queries
mloop agent "
Task: Handle missing values
Columns: Age (numeric), City (categorical)
Strategy: median for numeric, mode for categorical
" -a preprocessing-expert
```

**DON'T**:
```bash
# Too vague
mloop agent "analyze data" -a data-analyst

# Missing context
mloop agent "what's a good metric?" -a model-architect

# Too complex in one query
mloop agent "do everything" -a mlops-manager
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
# Streaming is automatic (same cost, better UX)
mloop agent "detailed analysis..." --agent data-analyst

# Use specific queries (fewer tokens)
mloop agent "missing value count per column" -a data-analyst

# Cache results locally
mloop agent "analyze data.csv" -a data-analyst > analysis.md
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
# Single query with specific agent
mloop agent "query" --agent <agent-name>
mloop agent "query" -a <agent-name>

# Auto-select agent based on query
mloop agent "query"

# Interactive conversation mode
mloop agent --interactive
mloop agent -i

# Interactive with specific agent
mloop agent --interactive --agent <agent-name>

# List all available agents
mloop agent --list-agents
mloop agent -l

# Help
mloop agent --help
```

### Workflow Commands

```bash
# Run a workflow definition
mloop agent workflow run <workflow-file.yaml>

# List available workflows
mloop agent workflow list

# Validate workflow syntax
mloop agent workflow validate <workflow-file.yaml>

# Check workflow execution status
mloop agent workflow status <execution-id>
```

### Agent Names

| Name | Description |
|------|-------------|
| `data-analyst` | Dataset analysis |
| `preprocessing-expert` | Script generation |
| `model-architect` | ML configuration |
| `mlops-manager` | Workflow orchestration |

### Examples

```bash
# Query with specific agent (--agent or -a)
mloop agent "analyze data" --agent data-analyst
mloop agent "generate script" -a preprocessing-expert

# Auto-select agent (no --agent flag)
mloop agent "What preprocessing is needed for train.csv?"

# Interactive mode for multi-turn conversations
mloop agent --interactive

# Complex query
mloop agent "train model on train.csv with target=Label time=300" -a mlops-manager
```

### Interactive Mode Commands

When in interactive mode (`mloop agent -i`):

| Command | Description |
|---------|-------------|
| `/agents` | List available agents |
| `/switch <name>` | Switch to specific agent |
| `/auto` | Enable auto agent selection |
| `/help` | Show available commands |
| `exit` or `quit` | End conversation |

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

**Version**: 1.1.0
**Last Updated**: 2024-12-30
**Related**: [AI-AGENT-ARCHITECTURE.md](AI-AGENT-ARCHITECTURE.md), [AI-AGENTS.md](AI-AGENTS.md)
