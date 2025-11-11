# AI Agents

MLoop provides AI-powered agents for interactive ML workflow assistance using Ironbees Agent Mode with multi-provider LLM support.

## Quick Start

```bash
# Set up LLM provider
export ANTHROPIC_API_KEY=sk-ant-your-key
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Chat with agent
mloop agent chat data-analyst "Analyze my dataset"
```

## Available Agents

| Agent | Purpose | Example |
|-------|---------|---------|
| `data-analyst` | Dataset analysis, ML readiness | "Analyze titanic.csv" |
| `preprocessing-expert` | Generate C# preprocessing scripts | "Handle missing values" |
| `model-architect` | Problem classification, AutoML config | "Recommend model for churn prediction" |
| `mlops-manager` | Workflow orchestration | "Train model on customer-data.csv" |

## LLM Providers

MLoop selects providers in priority order:

1. **GPUStack** (local) - Cost-effective, data privacy
2. **Anthropic Claude** (cloud) - Best quality
3. **Azure OpenAI** (enterprise) - Compliance
4. **OpenAI** (cloud) - Development

### Configuration

Create `.env` file (copy from `.env.example`):

```bash
# Option 1: GPUStack (Local, 89% cost savings)
GPUSTACK_ENDPOINT=http://localhost:8080/v1
GPUSTACK_API_KEY=your-key
GPUSTACK_MODEL=llama-3.1-8b

# Option 2: Anthropic (Recommended for production)
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Option 3: Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_MODEL=gpt-4o

# Option 4: OpenAI
OPENAI_API_KEY=sk-proj-your-key
OPENAI_MODEL=gpt-4o-mini
```

Only one provider is needed. MLoop automatically selects based on available credentials.

### Cost Comparison (per 1M tokens)

| Provider | Input | Output | Monthly* |
|----------|-------|--------|----------|
| GPUStack (Llama 3.1 8B) | ~$1 | ~$1 | $20-50 |
| Anthropic Claude 3.5 Sonnet | $3 | $15 | $300-600 |
| OpenAI GPT-4o-mini | $0.15 | $0.60 | $50-100 |
| Azure OpenAI GPT-4o | $10 | $30 | $800-1500 |

*Estimated for 10-20M tokens/month (medium usage)

## Usage Examples

### Data Analysis
```bash
mloop agent chat data-analyst "Analyze datasets/train.csv. What preprocessing is needed?"
```

**Response includes**:
- Dataset overview (rows, columns, types)
- Statistical summary
- Missing values and outliers
- ML readiness assessment
- Recommended preprocessing steps

### Preprocessing Scripts
```bash
mloop agent chat preprocessing-expert "Generate scripts to handle missing values in Age column"
```

**Response includes**:
- Complete C# code implementing `IPreprocessingScript`
- Sequential naming (01_handle_missing.cs)
- Ready to save to `.mloop/scripts/preprocess/`

### Model Configuration
```bash
mloop agent chat model-architect "Recommend AutoML settings for binary classification with 10K rows"
```

**Response includes**:
- Problem type classification
- Recommended time limit
- Performance metric selection
- Expected ML.NET trainers
- Complete `mloop train` command

### Workflow Orchestration
```bash
mloop agent chat mlops-manager "Train model on datasets/customer-churn.csv with target 'Churned'"
```

**Response includes**:
- Complete workflow plan
- CLI commands to execute
- Expected outputs
- Next steps

## Streaming Responses

For real-time feedback:

```bash
mloop agent stream data-analyst "Detailed analysis of large-dataset.csv"
```

## GPUStack Setup

For local deployment:

```bash
# Docker (requires GPU)
docker run -d -p 8080:8080 --gpus all gpustack/gpustack:latest

# Or pip install
pip install gpustack
gpustack start --port 8080
```

## Troubleshooting

**Error: "No LLM provider credentials found"**
- Solution: Create `.env` file with provider credentials

**Error: "Connection refused to GPUStack"**
- Solution: Verify GPUStack is running: `curl http://localhost:8080/health`

**Error: "Rate limit exceeded"**
- Solution: Switch to GPUStack (unlimited) or upgrade API plan

## Technical Details

**Architecture**: MLoop uses Ironbees Agent Mode with multi-provider infrastructure via `Microsoft.Extensions.AI`.

**Agent Implementation**: All agents extend `ConversationalAgent` base class with specialized system prompts.

**Provider Selection**: Environment variables are checked in priority order. First valid configuration is used.

**Note**: Requires Ironbees v0.1.5+ (ConversationalAgent feature). Currently using project references until release.

## Related Documentation

- [User Guide](GUIDE.md) - Core MLoop commands
- [Architecture](ARCHITECTURE.md) - Technical design
- `.env.example` - Complete configuration template
