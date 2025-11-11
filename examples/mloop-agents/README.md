# MLoop AI Agent Configuration Example

This directory contains a complete example of MLoop AI agent configuration with 4 specialized agents for conversational ML project management.

## 🤖 Available Agents

### 1. **data-analyst**
- **Purpose**: Dataset analysis and profiling
- **Capabilities**:
  - Statistical analysis
  - Feature distribution analysis
  - Data quality assessment
  - Missing value detection
  - Outlier identification
- **Use When**: You need to understand your dataset characteristics

### 2. **preprocessing-expert**
- **Purpose**: Generate ML.NET preprocessing scripts
- **Capabilities**:
  - Missing value handling
  - Categorical encoding
  - Feature normalization
  - Feature engineering
  - C# script generation
- **Use When**: You need to clean and transform your data

### 3. **model-architect**
- **Purpose**: Model selection and training strategy
- **Capabilities**:
  - Algorithm recommendation
  - Training configuration
  - Hyperparameter guidance
  - Performance optimization
  - Metric selection
- **Use When**: You need to choose the right ML.NET trainers

### 4. **mlops-manager**
- **Purpose**: ML project lifecycle orchestration
- **Capabilities**:
  - Workflow guidance
  - Experiment management
  - Deployment strategies
  - Production monitoring
  - Best practices
- **Use When**: You need help with the overall ML workflow

## 🚀 Quick Start

### Prerequisites

1. **Set up Azure OpenAI credentials**:
   ```bash
   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
   export AZURE_OPENAI_KEY="your-api-key"
   ```

2. **Build MLoop CLI** (from repository root):
   ```bash
   dotnet build src/MLoop.CLI/MLoop.CLI.csproj
   ```

### Using the Agents

#### Interactive Mode (Recommended)
```bash
# Navigate to this example directory
cd examples/mloop-agents

# Start interactive conversation
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent --interactive

# Available commands in interactive mode:
# /agents - List all available agents
# /switch data-analyst - Switch to specific agent
# /auto - Enable automatic agent selection
# /help - Show help
# exit - Quit
```

#### Single Query Mode
```bash
# Ask a specific question
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent "Analyze the Titanic dataset"

# Use a specific agent
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent \
  "Generate preprocessing script for missing values" \
  --agent preprocessing-expert
```

## 📖 Example Conversations

### Example 1: Dataset Analysis
```
You: Analyze my dataset at datasets/titanic.csv
data-analyst:
📊 Dataset Overview
- Rows: 891, Columns: 12
- Memory: ~150KB
- Target: Survived (binary classification)

🎯 Target Variable Analysis
- Class distribution: 38% survived, 62% died (imbalanced)
- Recommendation: Use F1-score or AUC as metric

🔍 Feature Insights
- Numerical: Age (177 missing), Fare (skewed distribution)
- Categorical: Sex (2 categories), Embarked (3 categories)
- High correlation: Pclass and Fare (-0.55)

⚠️ Data Quality Issues
- Missing values: Age (19%), Cabin (77%), Embarked (0.2%)
- Outliers detected: Fare (some extreme high values)

💡 Recommendations
1. Handle Age missing values with median imputation
2. Drop Cabin due to 77% missing
3. One-hot encode Sex and Embarked
4. Log-transform Fare to reduce skewness
```

### Example 2: Preprocessing Script Generation
```
You: Generate preprocessing script based on my data analysis
preprocessing-expert:
📋 Preprocessing Strategy
Based on your Titanic dataset analysis, I'll create a script that:
1. Imputes Age with median
2. Drops Cabin column
3. One-hot encodes categorical features
4. Log-transforms Fare

💻 Generated Script
[Complete C# code with ML.NET transformations]

📝 Usage Instructions
1. Save script to .mloop/scripts/preprocessing/titanic_preprocessing.cs
2. Run: mloop preprocess
3. Verify transformed data
```

### Example 3: Model Recommendation
```
You: What model should I use for this classification task?
model-architect:
🎯 Problem Summary
- Binary classification (imbalanced: 38/62)
- Dataset: 891 rows, 11 features
- Performance priority: F1-score

🏗️ Recommended Architecture
Primary: LightGBM
- Reasoning: Handles imbalance well, best performance for this size
- Expected F1-score: 0.75-0.82
- Training time: 2-3 minutes

Alternatives:
- FastTree: Faster, slightly lower accuracy
- FastForest: More robust to outliers

⚙️ Training Configuration
mloop train \
  --time 180 \
  --metric F1Score \
  --test-split 0.2
```

### Example 4: Complete Workflow Guidance
```
You: I have a dataset and want to build an ML model. What should I do?
mlops-manager:
📋 Current Status
- New project, no setup yet

🎯 Recommended Next Steps

1. **Initialize Project**
   mloop init titanic-classifier --task binary-classification
   cd titanic-classifier

2. **Organize Data**
   - Place CSV in datasets/train.csv
   - Edit mloop.yaml to set label_column: "Survived"

3. **Analyze Data**
   mloop agent "Analyze my training data" --agent data-analyst

4. **Generate Preprocessing**
   mloop agent "Create preprocessing script" --agent preprocessing-expert

5. **Train Model**
   mloop train --time 300 --metric F1Score

6. **Evaluate and Deploy**
   mloop evaluate
   mloop promote <best-experiment-id>

Would you like me to guide you through each step?
```

## 🎯 Agent Selection Guide

Use this flowchart to choose the right agent:

```
Start
  │
  ├─ Need to understand dataset? → data-analyst
  │
  ├─ Need to clean/transform data? → preprocessing-expert
  │
  ├─ Need to choose ML algorithm? → model-architect
  │
  ├─ Need workflow/project help? → mlops-manager
  │
  └─ Not sure? → Use --interactive mode with /auto
```

## 🛠️ Customizing Agents

You can modify agent behavior by editing the configuration files:

### Agent Configuration (`agent.yaml`)
```yaml
name: data-analyst
description: Your custom description
model:
  temperature: 0.3  # Lower = more deterministic
  maxTokens: 4000   # Response length limit
```

### System Prompt (`system-prompt.md`)
Edit the markdown file to:
- Add domain-specific knowledge
- Change response format
- Include company-specific guidelines
- Add custom examples

## 📚 Best Practices

1. **Start with data-analyst**: Always understand your data first
2. **Use interactive mode**: Better for exploratory workflows
3. **Combine agents**: Switch between agents as needed
4. **Save conversations**: Document important insights
5. **Iterate**: Use mlops-manager for continuous improvement

## 🔧 Troubleshooting

### Agent Not Found
```
Error: No suitable agent found
Solution: Check that .mloop/agents/ directory contains agent configurations
```

### Azure OpenAI Connection Error
```
Error: AZURE_OPENAI_ENDPOINT not set
Solution: Set environment variables before running
```

### Agent Response Too Generic
```
Solution: Provide more context in your question:
❌ "Analyze data"
✅ "Analyze the Titanic dataset for binary classification, focusing on missing values and class imbalance"
```

## 📖 Additional Resources

- [MLoop Documentation](../../README.md)
- [Ironbees Agent Framework](https://github.com/your-org/ironbees)
- [ML.NET Documentation](https://docs.microsoft.com/dotnet/machine-learning/)

## 🤝 Contributing

To add new agents:
1. Create directory: `.mloop/agents/your-agent-name/`
2. Add `agent.yaml` with configuration
3. Add `system-prompt.md` with instructions
4. Update this README

## 📄 License

This example is part of the MLoop project and follows the same license.
