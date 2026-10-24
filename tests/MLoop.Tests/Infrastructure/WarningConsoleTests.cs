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

    /// <summary>
    /// A collection keeps the warnings of its own async flow — the tasks it starts included — and nothing raised in
    /// another flow: a server trains a job beside other requests, and the job reports only its own warnings.
    /// </summary>
    [Fact]
    public async Task ACollection_KeepsItsOwnFlowsWarnings_AndNoOtherFlows()
    {
        var gate = new TaskCompletionSource();

        async Task<IReadOnlyList<string>> Job(string name)
        {
            using var collected = WarningConsole.Collect();
            await gate.Task;
            WarningConsole.Warn($"[bold]{name}[/] in the flow");
            await Task.Run(() => WarningConsole.Warn($"{name} in a task it started"));
            return collected.Warnings;
        }

        var a = Job("a");
        var b = Job("b");
        gate.SetResult();

        Assert.Equal(["a in the flow", "a in a task it started"], await a);
        Assert.Equal(["b in the flow", "b in a task it started"], await b);
    }

    [Fact]
    public void ACollection_StopsCollecting_WhenDisposed_AndAnOuterOneSeesTheInnersWarnings()
    {
        using var outer = WarningConsole.Collect();
        using (var inner = WarningConsole.Collect())
        {
            WarningConsole.Warn("inner");
            Assert.Equal(["inner"], inner.Warnings);
        }
        WarningConsole.Warn("after");

        Assert.Equal(["inner", "after"], outer.Warnings);
    }
}
