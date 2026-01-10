using System.ClientModel;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core.Orchestration;
using MLoop.Tests.Common;
using OpenAI;

namespace MLoop.AIAgent.Tests;

/// <summary>
/// Integration tests that require LLM API access.
/// These tests are EXCLUDED from CI/CD and only run locally with API keys configured.
///
/// To run these tests locally:
///   1. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable
///   2. Run: dotnet test --filter "Category=LLM"
///
/// Or run all tests including LLM tests:
///   ./scripts/test-all.ps1 (Windows)
///   ./scripts/test-all.sh (Linux/Mac)
/// </summary>
[Trait(TestCategories.Category, TestCategories.LLM)]
public class LLMIntegrationTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testDataFile;

    public LLMIntegrationTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_llm_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);

        // Create a simple test CSV file for ML task
        _testDataFile = Path.Combine(_testDirectory, "test_data.csv");
        File.WriteAllText(_testDataFile, @"age,income,education,employed
25,50000,bachelor,yes
30,60000,master,yes
35,70000,bachelor,yes
40,80000,phd,yes
28,55000,bachelor,no
32,65000,master,no
36,75000,phd,yes
45,90000,phd,yes
29,58000,bachelor,yes
33,68000,master,yes");
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
    public async Task OrchestrationService_WithRealLLM_ExecutesDataAnalysisPhase()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var sessionStore = new OrchestrationSessionStore(Path.Combine(_testDirectory, "orchestration"));
        var orchestrator = new MLOpsOrchestratorService(chatClient, sessionStore);

        var receivedEvents = new List<OrchestrationEvent>();

        // Act - Execute orchestration and collect events
        try
        {
            await foreach (var evt in orchestrator.ExecuteAsync(
                _testDataFile,
                new OrchestrationOptions
                {
                    TargetColumn = "employed",
                    TaskType = "BinaryClassification",
                    MaxTrainingTimeSeconds = 30
                },
                cancellationToken: new CancellationTokenSource(TimeSpan.FromMinutes(2)).Token))
            {
                receivedEvents.Add(evt);

                // Stop after data analysis phase completes
                if (evt is PhaseCompletedEvent phase && phase.PhaseName == "Data Analysis")
                {
                    break;
                }

                // Stop if any failure occurs
                if (evt is OrchestrationFailedEvent)
                {
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Timeout is acceptable for this test
        }

        // Assert - Verify we received expected events
        Assert.Contains(receivedEvents, e => e is OrchestrationStartedEvent);
        Assert.Contains(receivedEvents, e => e is PhaseStartedEvent phase && phase.PhaseName == "Data Analysis");

        // If we got to AgentStartedEvent, LLM integration is working
        var agentStarted = receivedEvents.OfType<AgentStartedEvent>().FirstOrDefault();
        if (agentStarted != null)
        {
            Assert.Equal("DataAnalyst", agentStarted.AgentName);
        }
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ChatClient_WithRealLLM_ReturnsValidResponse()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are a helpful ML assistant."),
            new(ChatRole.User, "What is binary classification?")
        };

        // Act
        var response = await chatClient.GetResponseAsync(messages);

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response.Messages);
        var assistantMessage = response.Messages.FirstOrDefault(m => m.Role == ChatRole.Assistant);
        Assert.NotNull(assistantMessage);
        Assert.NotEmpty(assistantMessage.Text);
        Assert.Contains("classification", assistantMessage.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ChatClient_WithRealLLM_StreamsCorrectly()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        var chatClient = CreateChatClient();
        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, "Count to 3.")
        };

        var chunks = new List<string>();

        // Act
        await foreach (var update in chatClient.GetStreamingResponseAsync(messages))
        {
            if (!string.IsNullOrEmpty(update.Text))
            {
                chunks.Add(update.Text);
            }
        }

        // Assert
        Assert.NotEmpty(chunks);
        var fullText = string.Concat(chunks);
        Assert.NotEmpty(fullText);
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

