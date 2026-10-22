using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Infrastructure;

public class WarningConsoleTests
{
    [Theory]
    [InlineData("⚠ Small dataset detected: 50 samples", "Small dataset detected: 50 samples")]
    [InlineData("⚠️  Some folds hold no minority rows", "Some folds hold no minority rows")]
    [InlineData("  ⚠ indented", "indented")]
    [InlineData("🟡 High class imbalance: 3:1", "🟡 High class imbalance: 3:1")]
    [InlineData("No sign at all", "No sign at all")]
    [InlineData("A ⚠ inside stays", "A ⚠ inside stays")]
    public void TheSeamsOwnPrefix_ReplacesAWarningSignTheMessageCarried(string message, string expected)
        => Assert.Equal(expected, WarningConsole.WithoutOwnWarningSign(message));
}
