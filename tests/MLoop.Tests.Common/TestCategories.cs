namespace MLoop.Tests.Common;

/// <summary>
/// Test category constants for xUnit Trait-based test filtering.
/// Use these to categorize tests for selective execution in CI/CD vs local development.
/// </summary>
public static class TestCategories
{
    /// <summary>
    /// Category trait name for xUnit.
    /// Usage: [Trait(TestCategories.Category, TestCategories.Unit)]
    /// </summary>
    public const string Category = "Category";

    /// <summary>
    /// Fast, isolated unit tests with no external dependencies.
    /// These run in CI/CD and locally.
    /// </summary>
    public const string Unit = "Unit";

    /// <summary>
    /// Integration tests that test component interactions.
    /// These run in CI/CD and locally (using mocks/in-memory resources).
    /// </summary>
    public const string Integration = "Integration";

    /// <summary>
    /// Tests requiring external LLM API calls (OpenAI, Azure OpenAI, etc.).
    /// These are SKIPPED in CI/CD, only run locally with API keys.
    /// </summary>
    public const string LLM = "LLM";

    /// <summary>
    /// Tests requiring database connections or file system heavy operations.
    /// These are SKIPPED in CI/CD, only run locally.
    /// </summary>
    public const string Database = "Database";

    /// <summary>
    /// Long-running tests (> 30 seconds).
    /// These are SKIPPED in CI/CD, only run locally.
    /// </summary>
    public const string Slow = "Slow";

    /// <summary>
    /// End-to-end tests requiring full system setup.
    /// These are SKIPPED in CI/CD, only run locally.
    /// </summary>
    public const string E2E = "E2E";
}

/// <summary>
/// Test environment detection utilities.
/// </summary>
public static class TestEnvironment
{
    /// <summary>
    /// Returns true if running in a CI environment (GitHub Actions, Azure DevOps, etc.)
    /// </summary>
    public static bool IsCI =>
        Environment.GetEnvironmentVariable("CI") == "true" ||
        Environment.GetEnvironmentVariable("GITHUB_ACTIONS") == "true" ||
        Environment.GetEnvironmentVariable("TF_BUILD") == "true" ||
        Environment.GetEnvironmentVariable("JENKINS_URL") != null;

    /// <summary>
    /// Returns true if LLM API keys are available for testing.
    /// </summary>
    public static bool HasLLMApiKeys =>
        !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("OPENAI_API_KEY")) ||
        !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY"));

    /// <summary>
    /// Skip message for CI-excluded tests.
    /// </summary>
    public const string CISkipMessage = "This test is excluded from CI/CD pipeline";

    /// <summary>
    /// Skip message for LLM-dependent tests.
    /// </summary>
    public const string LLMSkipMessage = "This test requires LLM API keys";
}
