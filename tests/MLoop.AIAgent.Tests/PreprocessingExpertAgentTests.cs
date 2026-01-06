using System.ClientModel;
using Ironbees.Core.Conversation;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Infrastructure;
using MLoop.Tests.Common;
using OpenAI;

namespace MLoop.AIAgent.Tests;

/// <summary>
/// Integration tests for preprocessing-expert agent with LLM.
/// These tests are EXCLUDED from CI/CD and only run locally with API keys configured.
///
/// To run these tests locally:
///   1. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable
///   2. Run: dotnet test --filter "Category=LLM"
/// </summary>
[Trait(TestCategories.Category, TestCategories.LLM)]
public class PreprocessingExpertAgentTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testAgentsDirectory;

    public PreprocessingExpertAgentTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_preprocessing_expert_test_{Guid.NewGuid():N}");
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
    public async Task PreprocessingExpert_DateTimeColumn_RecommendsFeatureExtraction()
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

        // Copy preprocessing-expert agent to test directory
        await CopyPreprocessingExpertAgentAsync();

        // Initialize orchestrator with test agents directory
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset has a column OrderDate with datetime values (2023-01-15 14:30:00).
Recommend feature engineering strategies for this datetime column.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "preprocessing-expert");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss datetime feature extraction
        var hasDateTimeFeatures =
            response.Contains("year", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("month", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("day_of_week", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("is_weekend", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasDateTimeFeatures, "Response should recommend datetime feature extraction");

        // Response should discuss cyclical encoding
        var hasCyclicalEncoding =
            response.Contains("sin", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("cos", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("cyclical", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasCyclicalEncoding, "Response should mention cyclical encoding for temporal features");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task PreprocessingExpert_HighCardinalityCategorical_RecommendsEncodingStrategies()
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

        await CopyPreprocessingExpertAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset has:
- City column: 250 unique values (high cardinality)
- Product category: 15 unique values (medium cardinality)

Recommend encoding strategies for these categorical features.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "preprocessing-expert");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss high cardinality handling
        var hasHighCardinalityStrategy =
            response.Contains("target encoding", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("frequency encoding", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("hash encoding", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasHighCardinalityStrategy, "Response should recommend strategies for high cardinality");

        // Response should distinguish between high and medium cardinality
        var hasEncodingDifferentiation =
            response.Contains("one-hot", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("label encoding", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasEncodingDifferentiation, "Response should differentiate encoding by cardinality level");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task PreprocessingExpert_MultipleFeatures_RecommendsInteractionFeatures()
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

        await CopyPreprocessingExpertAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Dataset has:
- Age (numeric, 18-75)
- Income (numeric, $20K-$500K)
- Region (categorical, 5 values: North, South, East, West, Central)

Recommend feature engineering strategies including interaction features.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "preprocessing-expert");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss interaction features
        var hasInteractionFeatures =
            response.Contains("interaction", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("combine", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("multiply", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasInteractionFeatures, "Response should recommend interaction features");

        // Response should discuss polynomial or ratio features
        var hasAdvancedFeatures =
            response.Contains("polynomial", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("squared", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("ratio", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasAdvancedFeatures, "Response should recommend polynomial or ratio features");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task PreprocessingExpert_FeatureEngineeringRequest_GeneratesScript()
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

        await CopyPreprocessingExpertAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"Generate a C# preprocessing script that:
1. Extracts year, month, day_of_week from OrderDate column
2. Creates is_weekend binary feature
3. Saves to output CSV

Use IPreprocessingScript interface.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "preprocessing-expert");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should contain C# code
        var hasCode =
            response.Contains("using", StringComparison.OrdinalIgnoreCase) &&
            response.Contains("class", StringComparison.OrdinalIgnoreCase) &&
            response.Contains("IPreprocessingScript", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasCode, "Response should contain C# code implementing IPreprocessingScript");

        // Response should have datetime feature extraction logic
        var hasDateTimeLogic =
            response.Contains("DateTime", StringComparison.OrdinalIgnoreCase) &&
            (response.Contains("Year", StringComparison.OrdinalIgnoreCase) ||
             response.Contains("Month", StringComparison.OrdinalIgnoreCase) ||
             response.Contains("DayOfWeek", StringComparison.OrdinalIgnoreCase));
        Assert.True(hasDateTimeLogic, "Response should include datetime feature extraction logic");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task PreprocessingExpert_EcommerceDomain_RecommendsDomainSpecificFeatures()
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

        await CopyPreprocessingExpertAgentAsync();
        await orchestrator.InitializeAsync(_testAgentsDirectory);

        var request = @"E-commerce dataset for customer churn prediction:
- Customer purchase history (dates, amounts)
- Product categories purchased
- Total order count
- Days since last purchase

Recommend domain-specific feature engineering for this e-commerce problem.";

        // Act
        var response = await orchestrator.ProcessAsync(request, "preprocessing-expert");

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response);

        // Response should discuss RFM or e-commerce specific features
        var hasDomainFeatures =
            response.Contains("recency", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("frequency", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("monetary", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("RFM", StringComparison.OrdinalIgnoreCase) ||
            response.Contains("average order", StringComparison.OrdinalIgnoreCase);
        Assert.True(hasDomainFeatures, "Response should recommend e-commerce domain-specific features");
    }

    /// <summary>
    /// Helper method to copy the preprocessing-expert agent to test directory.
    /// </summary>
    private async Task CopyPreprocessingExpertAgentAsync()
    {
        var sourceAgentDir = Path.Combine(
            Path.GetDirectoryName(typeof(IronbeesOrchestrator).Assembly.Location)!,
            "Agents",
            "preprocessing-expert");

        if (!Directory.Exists(sourceAgentDir))
        {
            // Try relative path from test project
            sourceAgentDir = Path.Combine(
                Directory.GetCurrentDirectory(),
                "..", "..", "..", "..", "..",
                "src", "MLoop.AIAgent",
                "Agents", "preprocessing-expert");
        }

        if (!Directory.Exists(sourceAgentDir))
        {
            throw new DirectoryNotFoundException($"Cannot find preprocessing-expert agent at {sourceAgentDir}");
        }

        var targetAgentDir = Path.Combine(_testAgentsDirectory, "preprocessing-expert");
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
