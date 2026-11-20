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

    private const string PidFileVariable = "MLOOP_TEST_JOB_PIDFILE";
    private const string HoldVariable = "MLOOP_TEST_JOB_HOLD";

    /// <summary>
    /// The claim the job exists for, with real processes: a process that holds a child in the job is killed outright (as <c>taskkill /F</c> does,
    /// which runs nothing in it) and the child goes with it. The control is the same arrangement without the job, where the child outlives
    /// its parent — the orphan this was written for — so the test can tell the two apart.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task A_parent_killed_outright_takes_a_held_child_with_it_and_leaves_an_unheld_one(bool held)
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Job objects are Windows only.");
        Assert.SkipWhen(Environment.GetEnvironmentVariable(PidFileVariable) is not null, "this is the helper process");

        var pidFile = Path.Combine(Path.GetTempPath(), "mloop-job-" + Guid.NewGuid().ToString("N") + ".pid");
        using var parent = StartHelper(pidFile, held);
        Process? child = null;
        try
        {
            for (var waited = 0; !File.Exists(pidFile) && waited < 600 && !parent.HasExited; waited++)
                await Task.Delay(50);
            Assert.True(File.Exists(pidFile), "the helper process did not start its child");
            // The file is written whole before the helper sleeps; read what is in it once it holds a number.
            int childId = 0;
            for (var tries = 0; tries < 100 && !int.TryParse(File.ReadAllText(pidFile).Trim(), out childId); tries++)
                await Task.Delay(50);
            child = Process.GetProcessById(childId);
            Assert.False(child.HasExited, "the child must be running before its parent is killed");

            parent.Kill();   // this process only — no tree: the child is not the parent's to end

            if (held)
                Assert.True(child.WaitForExit(10_000), "the child was still running after the process holding its job was killed");
            else
                Assert.False(child.WaitForExit(2_000), "without a job the child ended with its parent; the control does not show the orphan");
        }
        finally
        {
            if (child is { HasExited: false })
                child.Kill();
            child?.Dispose();
            try { File.Delete(pidFile); } catch (IOException) { }
        }
    }

    /// <summary>The helper the test above starts: runs a long-lived child, holds it in the job when asked, writes the child's id down and sleeps. Not a test of its own.</summary>
    [Fact]
    public async Task HelperProcess_RunsAChildAndSleeps()
    {
        var pidFile = Environment.GetEnvironmentVariable(PidFileVariable);
        Assert.SkipWhen(pidFile is null, "started by A_parent_killed_outright_takes_a_held_child_with_it_and_leaves_an_unheld_one");

        var child = LongLivedChild();
        using var job = Environment.GetEnvironmentVariable(HoldVariable) == "1" ? KillOnCloseJob.Hold(child) : null;
        await File.WriteAllTextAsync(pidFile!, child.Id.ToString(System.Globalization.CultureInfo.InvariantCulture));
        await Task.Delay(TimeSpan.FromMinutes(2));
    }

    private static Process StartHelper(string pidFile, bool held)
    {
        var entry = Environment.GetCommandLineArgs()[0];
        var self = Environment.ProcessPath!;
        var host = Path.GetFileNameWithoutExtension(self).Equals("dotnet", StringComparison.OrdinalIgnoreCase);
        var info = new ProcessStartInfo(self) { UseShellExecute = false, CreateNoWindow = true };
        if (host)
            info.ArgumentList.Add(entry);
        info.ArgumentList.Add("-method");
        info.ArgumentList.Add($"{typeof(KillOnCloseJobTests).FullName}.{nameof(HelperProcess_RunsAChildAndSleeps)}");
        info.Environment[PidFileVariable] = pidFile;
        info.Environment[HoldVariable] = held ? "1" : "0";
        return Process.Start(info)!;
    }
}
