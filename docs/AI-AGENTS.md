# AI Agents

MLoop provides AI-powered agents for interactive ML workflow assistance using Ironbees Agent Mode with multi-provider LLM support.

## Quick Start

```bash
# Set up LLM provider
export ANTHROPIC_API_KEY=sk-ant-your-key
export ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Query with specific agent
mloop agent "Analyze my dataset" --agent data-analyst

# Interactive mode
mloop agent --interactive

# Auto-select agent (based on query)
mloop agent "What preprocessing is needed for train.csv?"
```

## Available Agents

| Agent | Purpose | Example |
|-------|---------|---------|
| `data-analyst` | Dataset analysis, ML readiness | "Analyze titanic.csv" |
| `preprocessing-expert` | Generate C# preprocessing scripts | "Handle missing values" |
| `model-architect` | Problem classification, AutoML config | "Recommend model for churn prediction" |
| `mlops-manager` | Workflow orchestration | "Train model on customer-data.csv" |

---

## LLM Provider Configuration

MLoop automatically selects the LLM provider based on environment variables in priority order:

| Priority | Provider | Use Case | Environment Variables |
|----------|----------|----------|----------------------|
| 1 | **GPUStack** | Production self-hosted | `GPUSTACK_ENDPOINT`, `GPUSTACK_API_KEY` |
| 2 | **Anthropic** | Claude-specific features | `ANTHROPIC_API_KEY` |
| 3 | **Azure OpenAI** | Enterprise, compliance | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY` |
| 4 | **OpenAI** | Rapid prototyping | `OPENAI_API_KEY` |

### Complete .env Configuration

Create `.env` file in your project root (copy from `.env.example`):

```bash
# ========================================
# Option 1: GPUStack (Priority 1 - Recommended for Production)
# ========================================
# Self-hosted, 89% cost savings, data privacy
GPUSTACK_ENDPOINT=http://172.30.1.53:8080
GPUSTACK_API_KEY=gpustack_xxx
GPUSTACK_MODEL=llama-3.1-8b-instruct
# Models: llama-3.1-70b-instruct, llama-3.1-8b-instruct, mistral-7b-instruct, qwen2.5-14b-instruct

# ========================================
# Option 2: Anthropic (Priority 2 - Recommended for Development)
# ========================================
# Extended context (200K tokens), streaming, latest Claude models
ANTHROPIC_API_KEY=sk-ant-api03-xxx
ANTHROPIC_MODEL=claude-sonnet-4-20250514
# Models: claude-sonnet-4-20250514, claude-3-5-sonnet-20241022, claude-3-opus-20240229

# ========================================
# Option 3: Azure OpenAI (Priority 3)
# ========================================
# Enterprise deployments, compliance requirements
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
# Models: gpt-4o, gpt-4o-mini, gpt-4-turbo

# ========================================
# Option 4: OpenAI (Priority 4)
# ========================================
# Rapid prototyping, development
OPENAI_API_KEY=sk-proj-xxx
OPENAI_MODEL=gpt-4o-mini
# Models: gpt-4o, gpt-4o-mini, gpt-4-turbo
```

**Note**: Only one provider is needed. MLoop automatically selects based on available credentials.

### Cost Comparison (per 1M tokens)

| Provider | Input | Output | Monthly* |
|----------|-------|--------|----------|
| GPUStack (Llama 3.1 8B) | ~$1 | ~$1 | $20-50 |
| Anthropic Claude Sonnet 4.5 | $3 | $15 | $300-600 |
| OpenAI GPT-4o-mini | $0.15 | $0.60 | $50-100 |
| Azure OpenAI GPT-4o | $10 | $30 | $800-1500 |

*Estimated for 10-20M tokens/month (medium usage)

---

## Usage Examples

### Data Analysis
```bash
mloop agent "Analyze datasets/train.csv. What preprocessing is needed?" --agent data-analyst
```

**Response includes**:
- Dataset overview (rows, columns, types)
- Statistical summary
- Missing values and outliers
- ML readiness assessment
- Recommended preprocessing steps

### Preprocessing Scripts
```bash
mloop agent "Generate scripts to handle missing values in Age column" --agent preprocessing-expert
```

**Response includes**:
- Complete C# code implementing `IPreprocessingScript`
- Sequential naming (01_handle_missing.cs)
- Ready to save to `.mloop/scripts/preprocess/`

### Model Configuration
```bash
mloop agent "Recommend AutoML settings for binary classification with 10K rows" --agent model-architect
```

**Response includes**:
- Problem type classification
- Recommended time limit
- Performance metric selection
- Expected ML.NET trainers
- Complete `mloop train` command

### Workflow Orchestration
```bash
mloop agent "Train model on datasets/customer-churn.csv with target 'Churned'" --agent mlops-manager
```

**Response includes**:
- Complete workflow plan
- CLI commands to execute
- Expected outputs
- Next steps

---

## GPUStack Setup (Self-Hosted)

For local deployment with 89% cost savings:

```bash
# Docker (requires GPU)
docker run -d -p 8080:8080 --gpus all gpustack/gpustack:latest

# Or pip install
pip install gpustack
gpustack start --port 8080
```

See [Ironbees SELF_HOSTED_LLMS.md](https://github.com/iyulab/ironbees/blob/main/docs/SELF_HOSTED_LLMS.md) for detailed instructions.

---

## Advanced Configuration

### Temperature Control

All providers support temperature control (0.0 - 1.0):
- `0.0` - Deterministic, focused responses (recommended for ML tasks)
- `0.3-0.7` - Balanced creativity and consistency
- `1.0` - Maximum creativity and randomness

Currently fixed at `0.0` for all MLoop agents. To customize:

```yaml
# ~/.mloop/agents/your-agent/agent.yaml
temperature: 0.0  # Change this value
max_tokens: 4096  # Default, can increase to 8192, 16384
```

### Verify Active Provider

```bash
# Run with verbose logging
mloop agent "hello" --verbose

# Output shows:
# "Using GPUStack endpoint: ..."
# "Using Anthropic Claude API, model: ..."
```

---

## Troubleshooting

### "No LLM provider credentials found"
- **Solution**: Create `.env` file with at least one provider's credentials

### "Connection refused to GPUStack"
- **Solution**: Verify GPUStack is running: `curl http://localhost:8080/health`

### "Rate limit exceeded"
- **Solution**: Switch to GPUStack (unlimited) or upgrade API plan

### "Model not found"
- **GPUStack**: `curl http://your-endpoint/v1/models` (list deployed models)
- **Anthropic**: Use exact model IDs from documentation
- **Azure OpenAI**: Use deployment name (not model ID)
- **OpenAI**: Use model IDs from https://platform.openai.com/docs/models

### Wrong Provider Selected
Check priority order. Higher priority providers override lower ones:
1. GPUStack (if `GPUSTACK_ENDPOINT` + `GPUSTACK_API_KEY` set)
2. Anthropic (if `ANTHROPIC_API_KEY` set, no GPUStack)
3. Azure OpenAI (if `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` set)
4. OpenAI (if `OPENAI_API_KEY` set, fallback)

---

## Best Practices

### Production Deployments
1. **Use GPUStack for production** (cost, privacy, latency)
2. **Use Anthropic for development** (latest models, extended context)
3. **Set explicit model names** (avoid relying on defaults)
4. **Use temperature 0.0 for ML tasks** (deterministic results)

### Security
1. **Never commit API keys** (use .env files, add to .gitignore)
2. **Rotate keys regularly** (especially for production)
3. **Use environment-specific keys** (dev, staging, prod)

---

## Technical Details

**Architecture**: MLoop uses Ironbees Agent Mode with multi-provider infrastructure via `Microsoft.Extensions.AI`.

**Agent Implementation**: All agents extend `ConversationalAgent` base class with specialized system prompts stored in `~/.mloop/agents/`.

**Provider Selection**: Environment variables are checked in priority order. First valid configuration is used.

**Version Requirements**: Ironbees v0.4.1+ (AgentMode), .NET 10.0

---

## Related Documentation

- [User Guide](GUIDE.md) - Core MLoop commands
- [Architecture](ARCHITECTURE.md) - Technical design
- [Ironbees AgentMode](https://github.com/iyulab/ironbees) - Agent infrastructure
- `.env.example` - Complete configuration template
