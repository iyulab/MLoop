using System.Diagnostics;
using MLoop.CLI.Infrastructure;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// The stop <c>mloop serve</c> passes on to its server: on Unix a SIGTERM, which a process that does not handle it
/// ends on (exit 143) — the server's host handles it and shuts down gracefully, verified against a running server.
/// </summary>
public class StopSignalForwardingTests
{
    [Fact]
    public void Stop_ends_the_child_by_signal()
    {
        if (OperatingSystem.IsWindows())
            return; // No signal on Windows — the child is ended with its tree; Kill needs no test of ours.

        using var child = Process.Start(new ProcessStartInfo("sleep", "30") { UseShellExecute = false })!;

        StopSignalForwarding.Stop(child);

        Assert.True(child.WaitForExit(10_000), "the child did not end after the stop was forwarded");
        Assert.Equal(143, child.ExitCode); // 128 + SIGTERM: ended by the signal, not killed
    }

    [Fact]
    public void Stop_of_a_child_that_has_ended_does_nothing()
    {
        using var child = Process.Start(new ProcessStartInfo(OperatingSystem.IsWindows() ? "cmd" : "true",
            OperatingSystem.IsWindows() ? "/c exit 0" : "")
        { UseShellExecute = false })!;
        child.WaitForExit();

        StopSignalForwarding.Stop(child);
    }
}
