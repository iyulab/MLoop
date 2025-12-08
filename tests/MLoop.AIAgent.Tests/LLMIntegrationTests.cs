using MLoop.Tests.Common;

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
public class LLMIntegrationTests
{
    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task ProcessQuery_WithRealLLM_ReturnsValidResponse()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        // TODO: Add actual LLM integration test implementation

        // Act
        await Task.Delay(1); // Placeholder

        // Assert
        Assert.True(true, "LLM test placeholder");
    }

    [Fact(Skip = "Requires LLM API key - remove Skip to enable")]
    public async Task StreamResponse_WithRealLLM_StreamsCorrectly()
    {
        // Skip if no API keys available
        if (!TestEnvironment.HasLLMApiKeys)
        {
            Assert.Fail(TestEnvironment.LLMSkipMessage);
            return;
        }

        // Arrange
        // TODO: Add actual LLM streaming test implementation

        // Act
        await Task.Delay(1); // Placeholder

        // Assert
        Assert.True(true, "LLM streaming test placeholder");
    }
}

/// <summary>
/// Example of how to mark slow/heavy tests that should be excluded from CI.
/// </summary>
[Trait(TestCategories.Category, TestCategories.Slow)]
public class SlowTestsExample
{
    [Fact(Skip = "Example - demonstrates slow test category")]
    public async Task LongRunningOperation_CompletesSuccessfully()
    {
        // This test takes > 30 seconds and is excluded from CI
        await Task.Delay(TimeSpan.FromSeconds(30));
        Assert.True(true);
    }
}
