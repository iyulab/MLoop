using System.ClientModel;
using Ironbees.Core.Conversation;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Infrastructure;
using MLoop.Tests.Common;
using OpenAI;

namespace MLoop.AIAgent.Tests;

/// <summary>
/// Integration tests for data-analyst agent with LLM.
/// These tests are EXCLUDED from CI/CD and only run locally with API keys configured.
///
/// To run these tests locally:
///   1. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable
///   2. Run: dotnet test --filter "Category=LLM"
/// </summary>
[Trait(TestCategories.Category, TestCategories.LLM)]
public class DataAnalystAgentTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testAgentsDirectory;

    public DataAnalystAgentTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_data_analyst_test_{Guid.NewGuid():N}");
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
    public async Task DataAnalystAgent_AnalyzeDataset_ReturnsQualityAssessment()
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

        // Copy data-analyst agent to test directory
        await CopyDataAnalystAgentAsync();

        // Initialize orchestrator with test agents directory
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"I have a dataset with customer data:
- 10,000 rows
- CustomerID, Age, Income, City, Country, ChurnedYesNo
- Missing values: Age (5%), Income (10%)
- Income has outliers (some values >$1M)
- ChurnedYesNo: 8500 'No', 1500 'Yes'

What data quality issues should I address?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "data-analyst");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss missing values
        Assert.Contains("missing", response, StringComparison.OrdinalIgnoreCase);

        // Response should discuss outliers
        var hasOutlierMention =
            response.Contains("outlier", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("income", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasOutlierMention, "Response should address outlier handling");

        // Response should detect class imbalance
        var hasImbalanceMention =
            response.Contains("imbalanc", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("8500", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("1500", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasImbalanceMention, "Response should detect class imbalance");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task DataAnalystAgent_BinaryClassification_RecommendsTaskType()
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

        await CopyDataAnalystAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"I have a dataset with columns:
- CustomerID, Age, Income, Region
- Target column: ChurnedYesNo (values: 'Yes' or 'No')

What ML task type should I use?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "data-analyst");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should recommend binary classification
        var hasBinaryClassification =
            response.Contains("binary", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("classification", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasBinaryClassification, "Response should recommend binary classification");

        // Response should mention appropriate metrics
        var hasMetricRecommendation =
            response.Contains("F1", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("accuracy", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("AUC", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasMetricRecommendation, "Response should recommend appropriate metric");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task DataAnalystAgent_RegressionTask_RecommendsTaskType()
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

        await CopyDataAnalystAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset columns:
- Location, SquareMeters, Bedrooms, Bathrooms, YearBuilt
- Target: Price (continuous values from $100K to $5M)

What task type and metric should I use?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "data-analyst");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should recommend regression
        Assert.Contains("regression", response, StringComparison.OrdinalIgnoreCase);

        // Response should recommend regression metrics
        var hasRegressionMetric =
            response.Contains("RMSE", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("MAE", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("R-squared", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("R2", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasRegressionMetric, "Response should recommend regression metrics");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task DataAnalystAgent_PreprocessingStrategy_ProvidesRecommendations()
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

        await CopyDataAnalystAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Data issues:
- Age: 5% missing values (numeric)
- Category: 3% missing values (categorical, 50 unique values)
- Income: Outliers detected (max: $10M, median: $50K)

What preprocessing strategies should I use?";

        // Act
        var response = await orchestrator.ProcessAsync(request, "data-analyst");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss missing value strategies
        var hasMissingStrategy =
            response.Contains("imputation", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("mean", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("median", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("mode", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasMissingStrategy, "Response should provide missing value handling strategy");

        // Response should discuss outlier handling
        var hasOutlierStrategy =
            response.Contains("winsoriz", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("cap", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("log", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("transform", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasOutlierStrategy, "Response should provide outlier handling strategy");

        // Response should discuss encoding for high cardinality
        var hasEncodingStrategy =
            response.Contains("encoding", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("target", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("hash", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasEncodingStrategy, "Response should provide encoding strategy for high cardinality");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task DataAnalystAgent_ConversationHistory_MaintainsContext()
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

        await CopyDataAnalystAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var conversationId = Guid.NewGuid().ToString();

        // First message: Establish context about churn dataset
        var firstRequest = "I have a customer churn dataset with 10,000 rows and severe class imbalance (85% non-churn, 15% churn).";
        var firstResponse = await orchestrator.ProcessAsync(firstRequest, "data-analyst", conversationId);
        Assert.NotEmpty(firstResponse);

        // Second message: Follow-up that should reference previous context
        var secondRequest = "What metric should I use for evaluation?";
        var secondResponse = await orchestrator.ProcessAsync(secondRequest, "data-analyst", conversationId);

        // Assert
        Assert.NotNull(secondResponse);
        Assert.NotEmpty(secondResponse);

        // Response should provide metric recommendation considering imbalance
        var hasImbalanceAwareMetric =
            secondResponse.Contains("F1", StringComparison.OrdinalIgnoreCase) ||
            secondResponse.Contains("precision", StringComparison.OrdinalIgnoreCase) ||
            secondResponse.Contains("recall", StringComparison.OrdinalIgnoreCase) ||
            secondResponse.Contains("AUC", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasImbalanceAwareMetric, "Response should recommend metric suitable for imbalanced data");
    }

    /// <summary>
    /// Helper method to copy the data-analyst agent to test directory.
    /// </summary>
    private async Task CopyDataAnalystAgentAsync()
    {
        var sourceAgentDir = Path.Combine(
            Path.GetDirectoryName(typeof(IronbeesOrchestrator).Assembly.Location)!,
            "Agents",
            "data-analyst");

        if (!Directory.Exists(sourceAgentDir))
        {
            // Try relative path from test project
            sourceAgentDir = Path.Combine(
                Directory.GetCurrentDirectory(),
                "..", "..", "..", "..", "..",
                "src", "MLoop.AIAgent",
                "Agents", "data-analyst");
        }

        if (!Directory.Exists(sourceAgentDir))
        {
            throw new DirectoryNotFoundException($"Cannot find data-analyst agent at {sourceAgentDir}");
        }

        var targetAgentDir = Path.Combine(_testAgentsDirectory, "data-analyst");
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
