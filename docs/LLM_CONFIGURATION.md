# MLoop LLM Configuration Guide

**Version**: 1.0.0
**Ironbees AgentMode**: v0.1.5+
**Target Framework**: .NET 10.0

---

## Overview

MLoop uses Ironbees AgentMode for multi-provider LLM support. This guide covers all supported LLM configurations and their environment variables.

---

## Supported Providers

| Provider | Use Case | Priority | Cost |
|----------|----------|----------|------|
| **GPUStack** | Production self-hosted | 1 | Hardware only |
| **Anthropic** | Claude-specific features | 2 | Pay-per-use |
| **Azure OpenAI** | Enterprise, compliance | 3 | Pay-per-use |
| **OpenAI** | Rapid prototyping | 4 | Pay-per-use |

### Provider Selection Priority

MLoop automatically selects the LLM provider based on environment variables in this order:

1. **GPUStack** (OpenAI-Compatible) - if `GPUSTACK_ENDPOINT` and `GPUSTACK_API_KEY` are set
2. **Anthropic Claude** - if `ANTHROPIC_API_KEY` is set
3. **Azure OpenAI** - if `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_KEY` are set
4. **OpenAI** - if `OPENAI_API_KEY` is set

---

## 1. GPUStack (Self-Hosted) - Priority 1

**Use Case**: Production deployments, cost optimization, privacy-sensitive data, air-gapped environments

### Environment Variables

```bash
# Required
GPUSTACK_ENDPOINT=http://172.30.1.53:8080
GPUSTACK_API_KEY=gpustack_8ef8f2d1e0537fb8_9f99ccb2699267880f8a5787deab1cf1

# Optional (defaults to "default")
GPUSTACK_MODEL=llama-3.1-8b-instruct
```

### Complete .env Example

```bash
# GPUStack Configuration (Priority 1)
GPUSTACK_ENDPOINT=http://172.30.1.53:8080
GPUSTACK_API_KEY=gpustack_8ef8f2d1e0537fb8_9f99ccb2699267880f8a5787deab1cf1
GPUSTACK_MODEL=llama-3.1-70b-instruct

# Alternative models
# GPUSTACK_MODEL=llama-3.1-8b-instruct         # Faster, less capable
# GPUSTACK_MODEL=mistral-7b-instruct           # Balanced performance
# GPUSTACK_MODEL=qwen2.5-14b-instruct          # Chinese language support
```

### Supported Models (via GPUStack)

- `llama-3.1-70b-instruct` - Most capable, slower
- `llama-3.1-8b-instruct` - Balanced
- `mistral-7b-instruct` - Fast inference
- `qwen2.5-14b-instruct` - Multilingual
- Any model deployed in your GPUStack cluster

### Benefits

- 🔐 **Privacy**: Data stays on-premise
- 💰 **Cost**: No per-token charges (hardware cost only)
- 🚀 **Low Latency**: Local inference
- 🌐 **Offline**: Air-gapped operation
- 🔧 **Flexibility**: Use any open-source model

### Setup Guide

See [Ironbees SELF_HOSTED_LLMS.md](https://github.com/yourusername/ironbees/blob/main/docs/SELF_HOSTED_LLMS.md) for GPUStack installation instructions.

---

## 2. Anthropic Claude - Priority 2

**Use Case**: Extended context (200K tokens), streaming, latest Claude models

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional (defaults to Claude Sonnet 4.5)
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### Complete .env Example

```bash
# Anthropic Claude Configuration (Priority 2)
ANTHROPIC_API_KEY=sk-ant-api03-abc123def456...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Alternative models
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # Previous generation
# ANTHROPIC_MODEL=claude-3-opus-20240229      # Most capable Claude 3
# ANTHROPIC_MODEL=claude-3-haiku-20240307     # Fastest, most affordable
```

### Supported Models

- `claude-sonnet-4-20250514` - **Claude Sonnet 4.5** (latest, default)
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet
- `claude-3-opus-20240229` - Most capable Claude 3
- `claude-3-haiku-20240307` - Fastest Claude 3

### Features

- ✅ **Extended Context**: 200K tokens
- ✅ **Streaming**: Real-time token streaming
- ✅ **System Prompts**: Powerful instruction support
- ✅ **Temperature Control**: 0.0 - 1.0
- ⚠️ **Vision**: SDK supported (integration pending)
- ⚠️ **Function Calling**: SDK supported (integration pending)
- ⚠️ **Prompt Caching**: SDK supported (integration pending)

### Get API Key

1. Visit https://console.anthropic.com/
2. Create account or sign in
3. Go to API Keys section
4. Generate new key
5. Copy key and set `ANTHROPIC_API_KEY` environment variable

### Pricing

See https://www.anthropic.com/pricing

---

## 3. Azure OpenAI - Priority 3

**Use Case**: Enterprise deployments, compliance requirements, Microsoft ecosystem integration

### Environment Variables

```bash
# Required
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-azure-openai-key

# Optional (defaults to "gpt-4o")
AZURE_OPENAI_DEPLOYMENT=gpt-4o-deployment-name
```

### Complete .env Example

```bash
# Azure OpenAI Configuration (Priority 3)
AZURE_OPENAI_ENDPOINT=https://mloop-openai.openai.azure.com
AZURE_OPENAI_KEY=abc123def456...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-deployment

# Alternative deployments
# AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini-deployment  # Affordable, fast
# AZURE_OPENAI_DEPLOYMENT=gpt-4-turbo-deployment  # Previous generation
```

### Supported Models

- `gpt-4o` - Most capable multimodal
- `gpt-4o-mini` - Affordable and fast
- `gpt-4-turbo` - Previous generation
- `gpt-35-turbo` - Legacy model

### Features

- ✅ **Enterprise Security**: Azure compliance certifications
- ✅ **Regional Deployment**: Deploy in specific regions
- ✅ **Content Filtering**: Built-in safety filters
- ✅ **Reserved Capacity**: Guaranteed throughput

### Setup Guide

1. Create Azure OpenAI resource in Azure Portal
2. Deploy a model (e.g., gpt-4o)
3. Get endpoint URL and API key
4. Set environment variables

---

## 4. OpenAI - Priority 4

**Use Case**: Rapid prototyping, development, personal projects

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-proj-...

# Optional (defaults to "gpt-4o-mini")
OPENAI_MODEL=gpt-4o-mini
```

### Complete .env Example

```bash
# OpenAI Configuration (Priority 4)
OPENAI_API_KEY=sk-proj-abc123def456...
OPENAI_MODEL=gpt-4o-mini

# Alternative models
# OPENAI_MODEL=gpt-4o              # Most capable, more expensive
# OPENAI_MODEL=gpt-4-turbo         # Previous generation
# OPENAI_MODEL=gpt-3.5-turbo       # Legacy model
```

### Supported Models

- `gpt-4o` - Most capable multimodal
- `gpt-4o-mini` - Affordable and fast (default)
- `gpt-4-turbo` - Previous generation
- `gpt-3.5-turbo` - Legacy model

### Features

- ✅ **Latest Models**: Immediate access to new releases
- ✅ **Simple Auth**: Just API key needed
- ✅ **Pay-per-use**: No minimum commitment
- ✅ **No Infrastructure**: Fully managed

### Get API Key

1. Visit https://platform.openai.com/
2. Create account or sign in
3. Go to API Keys section
4. Create new secret key
5. Copy key and set `OPENAI_API_KEY` environment variable

### Pricing

See https://openai.com/pricing

---

## Advanced Configuration

### Temperature Control

All providers support temperature control (0.0 - 1.0):

- `0.0` - Deterministic, focused responses (recommended for ML tasks)
- `0.3-0.7` - Balanced creativity and consistency
- `1.0` - Maximum creativity and randomness

Currently fixed at `0.0` for all MLoop agents. To customize, modify agent YAML files:

```yaml
# ~/.mloop/agents/your-agent/agent.yaml
temperature: 0.0  # Change this value
```

### Token Limits

Default: `4096` tokens per response

To increase for specific agents:

```yaml
# ~/.mloop/agents/your-agent/agent.yaml
max_tokens: 8192  # or 16384, 32768
```

**Note**: Some models support up to 200K tokens (Claude), 128K tokens (GPT-4), or unlimited context (self-hosted).

---

## Configuration Testing

### Verify Active Provider

```bash
# Run any agent command with verbose logging
mloop agent "hello" --verbose

# Check logs for provider detection:
# "Using GPUStack endpoint: ..."
# "Using Anthropic Claude API, model: ..."
# "Using Azure OpenAI, deployment: ..."
# "Using OpenAI, model: ..."
```

### Test Provider Switching

```bash
# Test GPUStack (Priority 1)
export GPUSTACK_ENDPOINT=http://172.30.1.53:8080
export GPUSTACK_API_KEY=your-key
mloop agent "test"

# Test Anthropic (Priority 2) - remove GPUStack variables
unset GPUSTACK_ENDPOINT GPUSTACK_API_KEY
export ANTHROPIC_API_KEY=sk-ant-...
mloop agent "test"

# Test Azure OpenAI (Priority 3) - remove Anthropic
unset ANTHROPIC_API_KEY
export AZURE_OPENAI_ENDPOINT=https://...
export AZURE_OPENAI_KEY=your-key
mloop agent "test"

# Test OpenAI (Priority 4) - remove Azure
unset AZURE_OPENAI_ENDPOINT AZURE_OPENAI_KEY
export OPENAI_API_KEY=sk-proj-...
mloop agent "test"
```

---

## Troubleshooting

### Provider Not Detected

**Symptom**: "No LLM provider configured"

**Solution**: Ensure at least one set of provider environment variables is set:
- GPUStack: `GPUSTACK_ENDPOINT` + `GPUSTACK_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY`
- OpenAI: `OPENAI_API_KEY`

### Wrong Provider Selected

**Symptom**: Using unexpected provider (e.g., OpenAI instead of GPUStack)

**Solution**: Check priority order. Higher priority providers override lower ones:
1. GPUStack (if variables set)
2. Anthropic (if no GPUStack)
3. Azure OpenAI (if no GPUStack/Anthropic)
4. OpenAI (fallback)

### Authentication Errors

**GPUStack**: Verify endpoint is accessible: `curl http://172.30.1.53:8080/v1/models`
**Anthropic**: Verify API key format: `sk-ant-api03-...`
**Azure OpenAI**: Verify endpoint includes resource name: `https://your-resource.openai.azure.com`
**OpenAI**: Verify API key format: `sk-proj-...`

### Model Not Found

**Solution**: Check model name matches provider's available models:
- GPUStack: `curl http://your-endpoint/v1/models` (list deployed models)
- Anthropic: Use exact model IDs from documentation
- Azure OpenAI: Use deployment name (not model ID)
- OpenAI: Use model IDs from https://platform.openai.com/docs/models

---

## Migration Guide

### From Hardcoded Agents to File-Based

MLoop 1.0+ uses file-based agents in `~/.mloop/agents/`. To customize:

1. **List installed agents**:
   ```bash
   mloop agents list
   ```

2. **View agent location**:
   ```bash
   mloop agents info data-analyst
   ```

3. **Edit agent files**:
   ```bash
   # Windows
   notepad %USERPROFILE%\.mloop\agents\data-analyst\system-prompt.md
   notepad %USERPROFILE%\.mloop\agents\data-analyst\agent.yaml

   # Linux/Mac
   nano ~/.mloop/agents/data-analyst/system-prompt.md
   nano ~/.mloop/agents/data-analyst/agent.yaml
   ```

4. **Force update built-in agents** (overwrites modifications):
   ```bash
   mloop agents install --force
   ```

### From OpenAI to Self-Hosted

1. **Setup GPUStack** (see [SELF_HOSTED_LLMS.md](https://github.com/yourusername/ironbees/blob/main/docs/SELF_HOSTED_LLMS.md))

2. **Deploy a model** (e.g., Llama 3.1 8B)

3. **Update environment variables**:
   ```bash
   # Remove OpenAI
   unset OPENAI_API_KEY

   # Add GPUStack
   export GPUSTACK_ENDPOINT=http://172.30.1.53:8080
   export GPUSTACK_API_KEY=gpustack_xxx
   export GPUSTACK_MODEL=llama-3.1-8b-instruct
   ```

4. **Test**:
   ```bash
   mloop agent "test GPUStack integration"
   ```

---

## Best Practices

### Production Deployments

1. **Use GPUStack for production** (cost, privacy, latency)
2. **Use Anthropic for development** (latest models, extended context)
3. **Set explicit model names** (avoid relying on defaults)
4. **Monitor token usage** (especially for pay-per-use providers)
5. **Use temperature 0.0 for ML tasks** (deterministic results)

### Development Workflow

1. **Prototype with OpenAI/Anthropic** (fast iteration)
2. **Test with self-hosted** (cost validation)
3. **Deploy with GPUStack** (production environment)

### Security

1. **Never commit API keys** (use .env files, add to .gitignore)
2. **Rotate keys regularly** (especially for production)
3. **Use environment-specific keys** (dev, staging, prod)
4. **Monitor usage** (detect anomalies)

---

## References

- [Ironbees AgentMode Documentation](https://github.com/yourusername/ironbees/tree/main/docs)
- [Ironbees PROVIDERS.md](https://github.com/yourusername/ironbees/blob/main/docs/PROVIDERS.md)
- [Ironbees SELF_HOSTED_LLMS.md](https://github.com/yourusername/ironbees/blob/main/docs/SELF_HOSTED_LLMS.md)
- [Microsoft.Extensions.AI](https://learn.microsoft.com/en-us/dotnet/ai/how-to/chat-client)

---

## Changelog

### 1.0.0 (2025-01-XX)
- Initial release
- Multi-provider support: GPUStack, Anthropic, Azure OpenAI, OpenAI
- File-based agent system
- Environment variable configuration
- Updated to Claude Sonnet 4.5 (claude-sonnet-4-20250514)
