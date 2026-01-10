using System.ClientModel;
using Ironbees.Core.Conversation;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Infrastructure;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Tests.Common;
using OpenAI;

namespace MLoop.AIAgent.Tests;

/// <summary>
/// Integration tests for incremental-preprocessing agent with LLM.
/// These tests are EXCLUDED from CI/CD and only run locally with API keys configured.
///
/// To run these tests locally:
///   1. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable
///   2. Run: dotnet test --filter "Category=LLM"
/// </summary>
[Trait(TestCategories.Category, TestCategories.LLM)]
public class IncrementalPreprocessingAgentTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testAgentsDirectory;

    public IncrementalPreprocessingAgentTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_incremental_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);

        // Copy agent templates to test directory for agent loading
        _testAgentsDirectory = Path.Combine(_testDirectory, "agents");
        Directory.CreateDirectory(_testAgentsDirectory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDirectory))
        {
            try
            {
                Directory.Delete(_testDirectory, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
        GC.SuppressFinalize(this);
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task IncrementalPreprocessingAgent_ProcessRequest_ReturnsValidResponse()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var conversationStore = new FileSystemConversationStore(
            Path.Combine(_testDirectory, "conversations"),
            NullLogger<FileSystemConversationStore>.Instance);

        var orchestrator = new IronbeesOrchestrator(
            chatClient,
            new Ironbees.AgentMode.Configuration.LLMConfiguration
            {
                Model = "gpt-4o-mini",
                ApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty
            },
            conversationStore,
            NullLogger<IronbeesOrchestrator>.Instance);

        // Copy incremental-preprocessing agent to test directory
        await CopyIncrementalPreprocessingAgentAsync();

        // Initialize orchestrator with test agents directory
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"I have a large dataset (10GB) with customer data containing:
- Customer ID, Age, Income, City, Country
- Purchase history with 50+ product categories
- Some missing values in Income and City columns
- Suspected outliers in purchase amounts

I want to preprocess this data for a churn prediction model.
What sampling strategy and preprocessing rules would you recommend for the first stage?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "incremental-preprocessing");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss sampling strategy
        Assert.Contains("sampl", response, StringComparison.OrdinalIgnoreCase);

        // Response should address data quality issues
        var hasDataQualityMention =
            response.Contains("missing", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("outlier", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasDataQualityMention, "Response should address data quality issues");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task IncrementalPreprocessingAgent_ProgressiveStagingQuery_ExplainsStages()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var conversationStore = new FileSystemConversationStore(
            Path.Combine(_testDirectory, "conversations"),
            NullLogger<FileSystemConversationStore>.Instance);

        var orchestrator = new IronbeesOrchestrator(
            chatClient,
            new Ironbees.AgentMode.Configuration.LLMConfiguration
            {
                Model = "gpt-4o-mini",
                ApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty
            },
            conversationStore,
            NullLogger<IronbeesOrchestrator>.Instance);

        await CopyIncrementalPreprocessingAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = "Explain the progressive sampling stages you use for incremental preprocessing.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "incremental-preprocessing");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should mention stages
        Assert.Contains("stage", response, StringComparison.OrdinalIgnoreCase);

        // Response should mention increasing sample sizes
        var hasSamplingProgression =
            response.Contains("0.1%", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("progressive", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("incremental", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasSamplingProgression, "Response should explain progressive sampling");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task IncrementalPreprocessingAgent_RuleDiscoveryQuery_DiscussesPatterns()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var conversationStore = new FileSystemConversationStore(
            Path.Combine(_testDirectory, "conversations"),
            NullLogger<FileSystemConversationStore>.Instance);

        var orchestrator = new IronbeesOrchestrator(
            chatClient,
            new Ironbees.AgentMode.Configuration.LLMConfiguration
            {
                Model = "gpt-4o-mini",
                ApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty
            },
            conversationStore,
            NullLogger<IronbeesOrchestrator>.Instance);

        await CopyIncrementalPreprocessingAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"After analyzing the first 0.1% sample, I found:
- 5% missing values in 'age' column
- 15% missing values in 'email' column
- Values in 'status' column: 95% 'active', 3% 'inactive', 2% 'pending'
- Outliers detected in 'purchase_amount' (>$10,000)

What preprocessing rules should I apply?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "incremental-preprocessing");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss handling missing values
        Assert.Contains("missing", response, StringComparison.OrdinalIgnoreCase);

        // Response should discuss outlier handling
        var hasOutlierHandling =
            response.Contains("outlier", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("purchase_amount", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasOutlierHandling, "Response should discuss outlier handling");

        // Response should consider the imbalanced categorical distribution
        var hasImbalanceConsideration =
            response.Contains("status", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("95%", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("imbalanc", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasImbalanceConsideration, "Response should consider imbalanced status distribution");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task IncrementalPreprocessingAgent_HITLQuery_ExplainsCheckpoints()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var conversationStore = new FileSystemConversationStore(
            Path.Combine(_testDirectory, "conversations"),
            NullLogger<FileSystemConversationStore>.Instance);

        var orchestrator = new IronbeesOrchestrator(
            chatClient,
            new Ironbees.AgentMode.Configuration.LLMConfiguration
            {
                Model = "gpt-4o-mini",
                ApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty
            },
            conversationStore,
            NullLogger<IronbeesOrchestrator>.Instance);

        await CopyIncrementalPreprocessingAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = "When should I pause for human approval (HITL) during incremental preprocessing?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "incremental-preprocessing");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss HITL or human approval
        var hasHITLMention =
            response.Contains("HITL", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("human", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("approval", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("review", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasHITLMention, "Response should discuss human-in-the-loop checkpoints");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task IncrementalPreprocessingAgent_ConversationHistory_MaintainsContext()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var conversationStore = new FileSystemConversationStore(
            Path.Combine(_testDirectory, "conversations"),
            NullLogger<FileSystemConversationStore>.Instance);

        var orchestrator = new IronbeesOrchestrator(
            chatClient,
            new Ironbees.AgentMode.Configuration.LLMConfiguration
            {
                Model = "gpt-4o-mini",
                ApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty
            },
            conversationStore,
            NullLogger<IronbeesOrchestrator>.Instance);

        await CopyIncrementalPreprocessingAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var conversationId = Guid.NewGuid().ToString();

        // First message: Establish context about customer churn dataset
        var firstRequest = "I'm working on a customer churn prediction project with 1 million records.";
        var firstResponse = await orchestrator.ProcessAsync(firstRequest, "incremental-preprocessing", conversationId);
        Assert.NotEmpty(firstResponse);

        // Second message: Follow-up that should reference previous context
        var secondRequest = "What sample size should I use for stage 1?";
        var secondResponse = await orchestrator.ProcessAsync(secondRequest, "incremental-preprocessing", conversationId);

        // Assert
        Assert.NotNull(secondResponse);
        Assert.NotEmpty(secondResponse);

        // Response should provide sampling guidance (0.1% of 1M = ~1000 records)
        var hasSamplingGuidance =
            secondResponse.Contains("0.1%", StringComparison.OrdinalIgnoreCase) ||
            secondResponse.Contains("1000", StringComparison.OrdinalIgnoreCase) ||
            secondResponse.Contains("1,000", StringComparison.OrdinalIgnoreCase) ||
            secondResponse.Contains("stage", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasSamplingGuidance, "Response should provide sampling guidance referencing the dataset size");
    }

    /// <summary>
    /// Helper method to copy the incremental-preprocessing agent to test directory.
    /// </summary>
    private async Task CopyIncrementalPreprocessingAgentAsync()
    {
        var sourceAgentDir = Path.Combine(
            Path.GetDirectoryName(typeof(IronbeesOrchestrator).Assembly.Location)!,
            "AgentTemplates",
            "incremental-preprocessing");

        if (!Directory.Exists(sourceAgentDir))
        {
            // Try relative path from test project
            sourceAgentDir = Path.Combine(
                Directory.GetCurrentDirectory(),
                "..", "..", "..", "..", "..",
                "src", "MLoop.AIAgent",
                "AgentTemplates", "incremental-preprocessing");
        }

        if (!Directory.Exists(sourceAgentDir))
        {
            throw new DirectoryNotFoundException($"Cannot find incremental-preprocessing agent at {sourceAgentDir}");
        }

        var targetAgentDir = Path.Combine(_testAgentsDirectory, "incremental-preprocessing");
        Directory.CreateDirectory(targetAgentDir);

        // Copy agent.yaml
        var yamlSource = Path.Combine(sourceAgentDir, "agent.yaml");
        var yamlTarget = Path.Combine(targetAgentDir, "agent.yaml");
        File.Copy(yamlSource, yamlTarget, true);

        // Copy system-prompt.md
        var promptSource = Path.Combine(sourceAgentDir, "system-prompt.md");
        var promptTarget = Path.Combine(targetAgentDir, "system-prompt.md");

        if (File.Exists(promptSource))
        {
            File.Copy(promptSource, promptTarget, true);
        }
        else
        {
            // Create a basic system prompt if not found
            await File.WriteAllTextAsync(promptTarget, @"# Incremental Preprocessing Agent

You are an expert in incremental preprocessing for large datasets using progressive sampling.

## Capabilities
- Progressive sampling strategies (0.1% → 0.5% → 1.5% → 2.5% → 100%)
- Automatic rule discovery from samples
- Human-in-the-loop (HITL) integration for approval
- Convergence analysis to determine when rules are stable

## Guidelines
1. Start with minimal samples (0.1%) to quickly identify patterns
2. Progressively increase sample size to validate and refine rules
3. Suggest HITL checkpoints before bulk processing
4. Detect rule convergence to avoid unnecessary processing
5. Consider data quality, missing values, outliers, and imbalances

Provide practical, actionable preprocessing recommendations based on the user's data characteristics.");
        }
    }

    private static IChatClient CreateChatClient()
    {
        // Try OpenAI first
        var openAiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (!string.IsNullOrEmpty(openAiKey))
        {
            var openAiClient = new OpenAIClient(openAiKey);
            return openAiClient.GetChatClient("gpt-4o-mini").AsIChatClient();
        }

        // Try Azure OpenAI
        var azureKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");
        var azureEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        if (!string.IsNullOrEmpty(azureKey) && !string.IsNullOrEmpty(azureEndpoint))
        {
            var azureClient = new OpenAIClient(new ApiKeyCredential(azureKey), new OpenAIClientOptions
            {
                Endpoint = new Uri(azureEndpoint)
            });
            var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "gpt-4";
            return azureClient.GetChatClient(deploymentName).AsIChatClient();
        }

        throw new InvalidOperationException(
            "No LLM API key configured. Set OPENAI_API_KEY or AZURE_OPENAI_* environment variables.");
    }
}
