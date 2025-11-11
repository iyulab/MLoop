# Equipment Anomaly Detection - Complete MLOps Example

> **Real-world MLOps pipeline** demonstrating MLoop + FilePrepper + Ironbees integration for equipment anomaly detection in manufacturing.

## 🎯 Project Goal

Demonstrate a **production-ready MLOps workflow** that combines:
- **MLoop**: ML.NET AutoML framework for model training
- **FilePrepper**: Data preparation and validation
- **Ironbees AI Agents**: Intelligent assistance throughout the ML lifecycle

**Not just building a model** - showing the complete journey from raw data to production deployment.

## 📊 Dataset: 장비이상 조기탐지 (Equipment Anomaly Detection)

**Source**: ML-Resource/014-장비이상 조기탐지/Dataset

**Data Description**:
- **Sensor Data**: Temperature and current measurements from manufacturing equipment
- **Time Period**: September-October 2021
- **Measurement Frequency**: Every 5 seconds
- **Files**: 50+ daily CSV files + Error Lot List
- **Total Samples**: ~75,000 measurements
- **Features**:
  - `Process`: Equipment/process ID (1-42)
  - `Temp`: Temperature in Celsius
  - `Current`: Electrical current in Amperes
  - `Time`: Measurement timestamp
  - `Date`: Measurement date

**Challenge**: Detect equipment anomalies before failures occur

**Labels**: Created by joining sensor data with Error Lot List
- `IsError = 1`: Process appears in error list for that date
- `IsError = 0`: Normal operation

## 🚀 Quick Start

### Prerequisites

1. **.NET 9.0 SDK**
2. **LLM Provider** credentials (for AI agents):
   - **GPUStack** (local, recommended for on-premise)
   - **Anthropic Claude** (recommended for best AI quality)
   - **Azure OpenAI** (enterprise cloud)
   - **OpenAI** (development)
3. **Source dataset**: Path to ML-Resource/014-장비이상 조기탐지

### Setup

```bash
# 1. Navigate to project directory
cd examples/equipment-anomaly-detection

# 2. Configure .env file in project root (D:\data\MLoop\.env)
# The project automatically loads these environment variables (priority order):
#
# Option 1 - GPUStack (local OpenAI-compatible endpoint):
# GPUSTACK_ENDPOINT=http://172.30.1.53:8080
# GPUSTACK_API_KEY=gpustack_xxx
# GPUSTACK_MODEL=kanana-1.5
#
# Option 2 - Anthropic Claude (recommended for production):
# ANTHROPIC_API_KEY=sk-ant-xxx
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
#
# Option 3 - Azure OpenAI (enterprise cloud):
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
# AZURE_OPENAI_KEY=your-api-key
#
# Option 4 - OpenAI (development):
# OPENAI_API_KEY=sk-proj-xxx
# OPENAI_MODEL=gpt-4o-mini

# 3. Prepare data (FilePrepper integration)
./scripts/prepare-data.sh "../../ML-Resource/014-장비이상 조기탐지/Dataset/data/5공정_180sec"

# Output:
#   datasets/train.csv       (60% of data, earlier dates)
#   datasets/validation.csv  (20% of data, middle dates)
#   datasets/test.csv        (20% of data, recent dates)
```

### Run Complete Workflow

```bash
# Step 1: Data Analysis with AI Agent
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent \
  "Analyze datasets/train.csv for anomaly detection" \
  --agent data-analyst

# Step 2: Generate Preprocessing Script
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent \
  "Generate preprocessing script for this dataset" \
  --agent preprocessing-expert

# Step 3: Get Model Recommendations
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent \
  "Recommend best model for binary classification with 8% anomaly rate" \
  --agent model-architect

# Step 4: Train Model
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj train \
  --time 600 \
  --metric F1Score \
  --test-split 0.3

# Step 5: Evaluate Model
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj evaluate \
  --experiment <exp-id>

# Step 6: Promote to Production
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj promote <exp-id>
```

## 📚 Complete Documentation

### Detailed Workflow

See **[WORKFLOW.md](./WORKFLOW.md)** for the complete step-by-step MLOps pipeline including:
- Data preparation with FilePrepper
- AI-assisted data analysis
- Automated preprocessing script generation
- Model selection and training
- Evaluation and deployment
- Production monitoring

### File Structure

```
equipment-anomaly-detection/
├── .mloop/
│   └── agents/                    # AI agent configurations
│       ├── data-analyst/
│       ├── preprocessing-expert/
│       ├── model-architect/
│       └── mlops-manager/
│
├── raw-data/                      # Raw sensor CSV files (gitignored)
│   ├── kemp-abh-sensor-*.csv
│   └── Error Lot list.csv
│
├── datasets/                      # Prepared training data
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/                        # Trained models
│   ├── staging/                   # Experimental models
│   └── production/                # Production models
│
├── experiments/                   # Training experiments
│   └── exp-*/                     # Each experiment's artifacts
│
├── scripts/                       # Data preparation scripts
│   ├── prepare-data.sh            # Bash version
│   └── prepare-data.ps1           # PowerShell version
│
├── mloop.yaml                     # Project configuration
├── README.md                      # This file
└── WORKFLOW.md                    # Complete workflow guide
```

## 🔄 MLOps Pipeline Overview

```
┌───────────────────────────────────────────────────────────┐
│              Production MLOps Pipeline                     │
└───────────────────────────────────────────────────────────┘

Raw Data (50+ CSV files)
    │
    ├─[FilePrepper]─────────────────────────┐
    │  - Merge multiple files                │
    │  - Join with Error Lot List            │
    │  - Feature engineering                 │
    │  - Temporal train/val/test split       │
    └────────────────────────────────────────┘
    │
    ├─[Ironbees: data-analyst]──────────────┐
    │  - Statistical analysis                │
    │  - Class imbalance detection          │
    │  - Feature correlation analysis        │
    │  - Preprocessing recommendations       │
    └────────────────────────────────────────┘
    │
    ├─[Ironbees: preprocessing-expert]──────┐
    │  - Generate C# preprocessing script    │
    │  - Handle missing values               │
    │  - Encode categorical features         │
    │  - Normalize numerical features        │
    │  - Create time-series features         │
    └────────────────────────────────────────┘
    │
    ├─[Ironbees: model-architect]───────────┐
    │  - Recommend ML.NET trainers           │
    │  - Suggest training configuration      │
    │  - Predict performance                 │
    │  - Optimize for F1-Score               │
    └────────────────────────────────────────┘
    │
    ├─[MLoop: AutoML Training]──────────────┐
    │  - Explore multiple trainers           │
    │  - Optimize F1-Score metric            │
    │  - Track experiments                   │
    │  - Save best model                     │
    └────────────────────────────────────────┘
    │
    ├─[MLoop: Evaluation]───────────────────┐
    │  - Compute F1, Precision, Recall       │
    │  - Generate confusion matrix           │
    │  - Feature importance analysis         │
    │  - Production readiness check          │
    └────────────────────────────────────────┘
    │
    ├─[Ironbees: mlops-manager]─────────────┐
    │  - Interpret evaluation results        │
    │  - Recommend deployment strategy       │
    │  - Configure monitoring                │
    │  - Set up retraining triggers          │
    └────────────────────────────────────────┘
    │
    └─[Production Deployment]───────────────┐
       - Model serving (API or batch)       │
       - Monitoring dashboard                │
       - Drift detection                     │
       - Automated retraining                │
       └────────────────────────────────────┘
```

## 🎯 Key Features Demonstrated

### 1. FilePrepper Integration
✅ **Multi-file CSV merging** - Consolidate 50+ daily files
✅ **Data validation** - Catch encoding and schema issues
✅ **Complex joins** - Merge sensor data with error logs
✅ **Feature engineering** - Extract time-based features

### 2. Ironbees AI Agents
✅ **data-analyst** - Deep dataset insights and recommendations
✅ **preprocessing-expert** - Auto-generate C# preprocessing code
✅ **model-architect** - Intelligent trainer selection
✅ **mlops-manager** - End-to-end workflow orchestration

### 3. MLoop AutoML
✅ **Automated training** - Explore multiple ML.NET trainers
✅ **Experiment tracking** - Compare and select best model
✅ **Production deployment** - Model promotion and serving
✅ **Monitoring** - Track performance and detect drift

### 4. Time-Series Best Practices
✅ **Temporal splitting** - Train/val/test by date, not random
✅ **Rolling features** - Moving averages and statistics
✅ **Lag features** - Previous measurements
✅ **Rate of change** - Delta temperature and current

### 5. Imbalanced Classification
✅ **F1-Score optimization** - Appropriate metric for 8% anomaly rate
✅ **Class weighting** - Handle severe imbalance
✅ **Threshold tuning** - Balance precision and recall
✅ **Confusion matrix** - Detailed error analysis

## 📊 Expected Results

### Performance Targets

| Metric | Target | Expected Range | Notes |
|--------|--------|----------------|-------|
| **F1-Score** | >0.75 | 0.78-0.85 | Primary metric |
| **Precision** | >0.70 | 0.75-0.82 | Minimize false alarms |
| **Recall** | >0.70 | 0.80-0.88 | Catch most anomalies |
| **AUC** | >0.85 | 0.88-0.92 | Discrimination ability |
| **Latency** | <100ms | 20-50ms | Production inference |

### Training Performance

- **Best Trainer**: LightGBM
- **Training Time**: 8-12 minutes (600-second budget)
- **Model Size**: 5-8 MB
- **Feature Importance**: Temp_Rolling_Mean, Current_Lag_1, Temp × Current

## 🛠️ Advanced Usage

### Interactive AI Agent Mode

```bash
# Start interactive session
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent --interactive

# Available commands:
/agents              # List all AI agents
/switch <agent>      # Switch to specific agent
/auto                # Auto-select agent
/help                # Show help
exit                 # Quit

# Example conversation:
You: Analyze my dataset
[agent auto-selects data-analyst]

data-analyst: [Provides detailed analysis]

You: /switch preprocessing-expert
You: Generate preprocessing script based on that analysis

preprocessing-expert: [Generates C# code]

You: /switch model-architect
You: What's the best model for this?

model-architect: [Recommends LightGBM with config]
```

### Custom Preprocessing

```bash
# Add your own preprocessing script
# 1. Create script in .mloop/scripts/preprocessing/
# 2. Implement IPreprocessingScript interface
# 3. MLoop auto-discovers and runs it

# Example: .mloop/scripts/preprocessing/custom_features.cs
using MLoop.Extensibility;

public class CustomFeatures : IPreprocessingScript
{
    public string Name => "Custom Time-Series Features";

    public IDataView Transform(MLContext mlContext, IDataView dataView)
    {
        // Your custom feature engineering logic
        return dataView;
    }
}
```

### Production Deployment

```bash
# Option 1: Real-time API serving
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj serve --port 8080

# API endpoints:
# POST /predict        - Single prediction
# POST /predict/batch  - Batch predictions
# GET /health          - Health check
# GET /metrics         - Model metrics

# Option 2: Batch predictions
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj predict \
  --input new-sensor-data.csv \
  --output predictions.csv \
  --threshold 0.5
```

## 🔍 Troubleshooting

### Data Preparation Issues

**Problem**: "Source path not found"
```bash
# Solution: Update path in prepare-data.sh
./scripts/prepare-data.sh "path/to/your/data/5공정_180sec"
```

**Problem**: "Encoding error when reading CSV"
```bash
# Solution: Data has UTF-8 BOM, script handles this automatically
# If issues persist, use FilePrepper for robust encoding detection
```

### Model Training Issues

**Problem**: "Low F1-Score (<0.70)"
```
Possible causes:
1. Insufficient training time → Increase --time to 900
2. Poor features → Review preprocessing script
3. Severe imbalance → Adjust class weights or threshold
4. Data leakage → Check temporal split

Ask mlops-manager agent:
"My F1-score is only 0.65, what should I investigate?"
```

**Problem**: "Overfitting (train F1 >> validation F1)"
```
Solutions:
1. Reduce training time
2. Use cross-validation
3. Add regularization
4. Simplify features
```

### Agent Issues

**Problem**: "No suitable agent found"
```bash
# Solution: Check agent configurations exist
ls .mloop/agents/*/agent.yaml

# Re-copy if needed:
cp -r ../mloop-agents/.mloop/agents/* .mloop/agents/
```

**Problem**: "LLM provider connection error"
```bash
# Solution: Verify .env file in D:\data\MLoop\.env
cat D:\data\MLoop\.env

# Priority order (only ONE set needed):
# 1. GPUStack (local):
# GPUSTACK_ENDPOINT=http://172.30.1.53:8080
# GPUSTACK_API_KEY=gpustack_xxx

# 2. Anthropic Claude (production):
# ANTHROPIC_API_KEY=sk-ant-xxx

# 3. Azure OpenAI (enterprise):
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
# AZURE_OPENAI_KEY=your-api-key

# 4. OpenAI (development):
# OPENAI_API_KEY=sk-proj-xxx
```

## 📚 Learn More

- **[WORKFLOW.md](./WORKFLOW.md)** - Complete step-by-step MLOps pipeline
- **[../../README.md](../../README.md)** - MLoop framework documentation
- **[../mloop-agents/README.md](../mloop-agents/README.md)** - AI agent usage guide

## 🤝 Contributing

This example demonstrates best practices for:
- Real-world MLOps workflows
- FilePrepper integration for data preparation
- AI-assisted ML development with Ironbees
- Production-ready model deployment

To extend this example:
1. Add more sophisticated feature engineering
2. Implement online learning for concept drift
3. Create monitoring dashboard
4. Add A/B testing framework
5. Integrate with CI/CD pipeline

## 📄 License

Part of the MLoop project. See main repository for license details.

---

**Built with**: MLoop + FilePrepper + Ironbees
**Dataset**: 장비이상 조기탐지 (Equipment Anomaly Detection)
**Purpose**: Demonstrate production-ready MLOps workflows
