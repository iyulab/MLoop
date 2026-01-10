using System.ClientModel;
using Ironbees.Core.Conversation;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Infrastructure;
using MLoop.Tests.Common;
using OpenAI;

namespace MLoop.AIAgent.Tests;

/// <summary>
/// Integration tests for model-architect agent with LLM.
/// These tests are EXCLUDED from CI/CD and only run locally with API keys configured.
///
/// To run these tests locally:
///   1. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable
///   2. Run: dotnet test --filter "Category=LLM"
/// </summary>
[Trait(TestCategories.Category, TestCategories.LLM)]
public class ModelArchitectAgentTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testAgentsDirectory;

    public ModelArchitectAgentTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_model_architect_test_{Guid.NewGuid():N}");
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
    public async Task ModelArchitectAgent_SimpleDataset_RecommendsBaseTimeBudget()
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

        // Copy model-architect agent to test directory
        await CopyModelArchitectAgentAsync();

        // Initialize orchestrator with test agents directory
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset characteristics:
- 500 rows
- 5 features (all numeric, no missing values)
- Target: ChurnedYesNo (binary classification, balanced)

Recommend AutoML configuration.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "model-architect");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss time budget
        var hasTimeBudget =
            response.Contains("time", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("second", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("60", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("120", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasTimeBudget, "Response should provide time budget recommendation");

        // Response should recommend appropriate metric
        var hasMetric =
            response.Contains("accuracy", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("F1", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasMetric, "Response should recommend metric");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ModelArchitectAgent_ComplexDataset_AdjustsTimeBudget()
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

        await CopyModelArchitectAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset characteristics:
- 5,000 rows
- 75 features (mix of numeric and categorical)
- 15% missing values in 10 features
- High cardinality in 3 categorical features (>100 unique values each)
- Target: Churn (binary classification, severe imbalance 1:10)

Recommend AutoML configuration with complexity adjustments.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "model-architect");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss complexity factors
        var hasComplexityDiscussion =
            response.Contains("complex", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("feature", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("adjust", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("multiplier", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasComplexityDiscussion, "Response should discuss complexity factors");

        // Response should recommend longer time for complex dataset
        var hasIncreasedTime =
            response.Contains("300", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("400", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("500", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("600", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasIncreasedTime, "Response should recommend increased time budget for complex dataset");

        // Response should recommend imbalance-aware metric
        var hasImbalanceMetric =
            response.Contains("F1", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("AUC", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("imbalanc", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasImbalanceMetric, "Response should recommend metric suitable for imbalanced data");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ModelArchitectAgent_BinaryClassification_RecommendsConfiguration()
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

        await CopyModelArchitectAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Problem: Predict customer churn (Yes/No)
Dataset: 10,000 rows, 20 features, balanced classes
Goal: Maximize F1-score

Provide AutoML configuration.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "model-architect");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should classify as binary classification
        var hasBinaryClassification =
            response.Contains("binary", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("classification", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasBinaryClassification, "Response should identify binary classification");

        // Response should provide mloop command
        var hasCommand =
            response.Contains("mloop", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("train", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasCommand, "Response should provide MLoop command");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ModelArchitectAgent_RegressionTask_RecommendsConfiguration()
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

        await CopyModelArchitectAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Problem: Predict house prices (continuous numeric)
Dataset: 15,000 rows, 30 features (location, size, amenities)
Goal: Minimize prediction error

Provide AutoML configuration.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "model-architect");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should identify regression
        Assert.Contains("regression", response, StringComparison.OrdinalIgnoreCase);

        // Response should recommend regression metric
        var hasRegressionMetric =
            response.Contains("RMSE", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("MAE", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("R-Squared", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("R2", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasRegressionMetric, "Response should recommend regression metric");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ModelArchitectAgent_ConfigurationExplanation_ProvidesRationale()
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

        await CopyModelArchitectAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset: 8,000 rows, 50 features, multiclass (5 classes)
Missing values: 10%
Class distribution: Imbalanced (40%, 30%, 15%, 10%, 5%)

Explain the time budget and metric recommendations.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "model-architect");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should provide reasoning
        var hasReasoning =
            response.Contains("reason", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("because", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("explain", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasReasoning, "Response should provide reasoning for recommendations");

        // Response should discuss multiclass considerations
        var hasMulticlassDiscussion =
            response.Contains("multiclass", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("class", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasMulticlassDiscussion, "Response should discuss multiclass considerations");
    }

    /// <summary>
    /// Helper method to copy the model-architect agent to test directory.
    /// </summary>
    private async Task CopyModelArchitectAgentAsync()
    {
        var sourceAgentDir = Path.Combine(
            Path.GetDirectoryName(typeof(IronbeesOrchestrator).Assembly.Location)!,
            "Agents",
            "model-architect");

        if (!Directory.Exists(sourceAgentDir))
        {
            // Try relative path from test project
            sourceAgentDir = Path.Combine(
                Directory.GetCurrentDirectory(),
                "..", "..", "..", "..", "..",
                "src", "MLoop.AIAgent",
                "Agents", "model-architect");
        }

        if (!Directory.Exists(sourceAgentDir))
        {
            throw new DirectoryNotFoundException($"Cannot find model-architect agent at {sourceAgentDir}");
        }

        var targetAgentDir = Path.Combine(_testAgentsDirectory, "model-architect");
        Directory.CreateDirectory(targetAgentDir);

        // Copy agent.yaml
        var yamlSource = Path.Combine(sourceAgentDir, "agent.yaml");
        var yamlTarget = Path.Combine(targetAgentDir, "agent.yaml");
        File.Copy(yamlSource, yamlTarget, true);

        // Copy system-prompt.md
        var promptSource = Path.Combine(sourceAgentDir, "system-prompt.md");
        var promptTarget = Path.Combine(targetAgentDir, "system-prompt.md");
        File.Copy(promptSource, promptTarget, true);
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
