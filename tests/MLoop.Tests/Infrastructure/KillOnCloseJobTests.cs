using System.Diagnostics;
using System.Runtime.Versioning;
using MLoop.CLI.Infrastructure;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// The job <c>mloop serve</c> puts its server in on Windows: the system ends the server when the parent's handle on the job closes — which
/// is what happens when the parent is killed outright. Windows only; there is nothing to hold on another system, so each test says so
/// and skips there.
/// </summary>
[SupportedOSPlatform("windows")]
public class KillOnCloseJobTests
{
    // `ping -n 120` answers once a second for two minutes: a child that lives long enough to be ended, and starts nothing of its own.
    private static Process LongLivedChild() =>
        Process.Start(new ProcessStartInfo("ping", "-n 120 127.0.0.1")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
        })!;

    [Fact]
    public void A_child_in_the_job_ends_when_the_handle_closes()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Job objects are Windows only.");

        using var child = LongLivedChild();
        try
        {
            using var job = KillOnCloseJob.Hold(child);
            Assert.NotNull(job);
            Assert.False(child.HasExited, "the child must be running while the job is held");

            job.Dispose();

            Assert.True(child.WaitForExit(10_000), "the child was still running after the job's handle closed");
        }
        finally
        {
            if (!child.HasExited)
                child.Kill();
        }
    }

    [Fact]
    public void A_child_outside_a_job_is_not_ended_by_one_that_closed()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Job objects are Windows only.");

        using var held = LongLivedChild();
        using var other = LongLivedChild();
        try
        {
            using var job = KillOnCloseJob.Hold(held);
            Assert.NotNull(job);

            job.Dispose();

            Assert.True(held.WaitForExit(10_000));
            Assert.False(other.HasExited, "closing one job ended a process that was never in it");
        }
        finally
        {
            if (!held.HasExited)
                held.Kill();
            if (!other.HasExited)
                other.Kill();
        }
    }

    [Fact]
    public void A_child_that_has_already_ended_is_nothing_to_hold()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Job objects are Windows only.");

        using var child = Process.Start(new ProcessStartInfo("cmd", "/c exit 0") { UseShellExecute = false, CreateNoWindow = true })!;
        child.WaitForExit();

        using var job = KillOnCloseJob.Hold(child);

        // Either the system refuses (null) or the job holds nothing; what must not happen is an exception into the serve command.
        job?.Dispose();
    }
}
