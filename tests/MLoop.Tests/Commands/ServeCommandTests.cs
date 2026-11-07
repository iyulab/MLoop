using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using MLoop.CLI.Commands;
using Xunit;

namespace MLoop.Tests.Commands;

/// <summary>
/// D9: dev-build API assembly resolution must prefer the most recently built DLL so a fresh
/// Release build wins over a stale Debug one. The old "first existing, Debug before Release"
/// order loaded the stale Debug assembly and hid Release-only fixes (which is why this has to set
/// MLOOP_API_PATH=Release to work around it).
/// </summary>
public class ServeCommandTests
{
    [Fact]
    public void NewestExisting_PrefersMostRecentlyBuilt()
    {
        var times = new Dictionary<string, DateTime>
        {
            ["Debug/MLoop.API.dll"] = new DateTime(2026, 1, 1),
            ["Release/MLoop.API.dll"] = new DateTime(2026, 6, 30), // newer build
        };

        var result = ServeCommand.NewestExisting(
            new[] { "Debug/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: _ => true,
            lastWriteUtc: p => times[p]);

        Assert.Equal("Release/MLoop.API.dll", result);
    }

    [Fact]
    public void NewestExisting_DebugNewer_PicksDebug()
    {
        // Symmetry: the rule is "newest", not "always Release" — a fresher Debug build wins too.
        var times = new Dictionary<string, DateTime>
        {
            ["Debug/MLoop.API.dll"] = new DateTime(2026, 6, 30),
            ["Release/MLoop.API.dll"] = new DateTime(2026, 1, 1),
        };

        var result = ServeCommand.NewestExisting(
            new[] { "Debug/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: _ => true,
            lastWriteUtc: p => times[p]);

        Assert.Equal("Debug/MLoop.API.dll", result);
    }

    [Fact]
    public void NewestExisting_SkipsMissingCandidates()
    {
        var result = ServeCommand.NewestExisting(
            new[] { "ghost/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: p => p.StartsWith("Release"),
            lastWriteUtc: _ => new DateTime(2026, 6, 30));

        Assert.Equal("Release/MLoop.API.dll", result);
    }

    [Fact]
    public void NewestExisting_NoneExist_ReturnsNull()
    {
        var result = ServeCommand.NewestExisting(
            new[] { "Debug/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: _ => false,
            lastWriteUtc: _ => DateTime.UtcNow);

        Assert.Null(result);
    }

    [Theory]
    [InlineData("localhost")]
    [InlineData("127.0.0.1")]
    [InlineData("0.0.0.0")]
    [InlineData("::1")]
    [InlineData("[::1]")]
    [InlineData("api.example.com")]
    [InlineData("*")]
    [InlineData("+")]
    public void A_host_name_or_address_is_bound(string host) => Assert.True(ServeCommand.IsBindableHost(host));

    /// <summary>The host is spliced into the API's command line, so a value that carries arguments is refused.</summary>
    [Theory]
    [InlineData("127.0.0.1 --Jwt:Key=0123456789abcdef0123456789abcdef")]
    [InlineData("localhost\" --urls http://0.0.0.0:80 \"")]
    [InlineData("")]
    [InlineData("a b")]
    public void A_value_that_is_not_a_host_is_refused(string host) => Assert.False(ServeCommand.IsBindableHost(host));
    // ---- a server in the background holds nothing of the caller's ----

    /// <summary>
    /// The load-independent half: the shape of the start. The program runs behind a shell that gives it a log for its
    /// output and nothing for its input, and every argument reaches it whole — a path with a space in it included.
    /// </summary>
    [Fact]
    public void A_detached_start_goes_through_a_shell_that_takes_the_output_and_input_away()
    {
        var info = ServeCommand.DetachedStartInfo(
            "dotnet", ["/opt/my tools/MLoop.API.dll", "--urls", "http://localhost:5001"], "/p/.mloop/temp/serve-5001.log", "/p");

        Assert.Equal("/bin/sh", info.FileName);
        Assert.False(info.UseShellExecute);
        Assert.False(info.RedirectStandardOutput);
        Assert.Equal("/p", info.WorkingDirectory);

        var args = info.ArgumentList;
        Assert.Contains(">>\"$log\" 2>&1 </dev/null", args[1], StringComparison.Ordinal);
        Assert.Equal(
            new[] { "/p/.mloop/temp/serve-5001.log", "dotnet", "/opt/my tools/MLoop.API.dll", "--urls", "http://localhost:5001" },
            args.Skip(3).ToArray());
    }

    /// <summary>
    /// The behaviour: a caller reading the command's output (a pipe, as <c>mloop serve --detach | tee</c> or a CI step does)
    /// reaches the end of it while the server is still running, and what the server writes lands in the log.
    /// The wait is a safety cap for a fault, not a speed claim: the end of the output is not a thing time decides.
    /// </summary>
    [Fact]
    public async Task A_detached_server_leaves_the_callers_pipe_open_for_no_one_and_writes_to_its_log()
    {
        Assert.SkipWhen(OperatingSystem.IsWindows(), "the shell start is Unix only");

        var dir = Path.Combine(Path.GetTempPath(), "mloop-detach-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var log = Path.Combine(dir, "serve.log");
        Process? server = null;
        try
        {
            var info = ServeCommand.DetachedStartInfo(
                "/bin/sh", ["-c", "echo to-stdout; echo to-stderr >&2; exec sleep 600"], log, dir);
            // What a caller reading the command's output gives its children: the write end of a pipe.
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;

            server = Process.Start(info)!;
            var stdoutEnd = server.StandardOutput.ReadToEndAsync();
            var stderrEnd = server.StandardError.ReadToEndAsync();

            var finished = await Task.WhenAny(Task.WhenAll(stdoutEnd, stderrEnd), Task.Delay(TimeSpan.FromSeconds(60)));
            Assert.True(
                stdoutEnd.IsCompleted && stderrEnd.IsCompleted,
                "the caller's pipe is still open: the server holds it");
            Assert.False(server.HasExited, "the server must still be running");
            Assert.Equal("", await stdoutEnd);
            Assert.Equal("", await stderrEnd);

            // The shell wrote nothing to the pipe; the program's words are in the log (give the process a moment to write them).
            string text = "";
            for (var i = 0; i < 600 && !(text.Contains("to-stdout") && text.Contains("to-stderr")); i++)
            {
                text = File.Exists(log) ? await File.ReadAllTextAsync(log) : "";
                if (!(text.Contains("to-stdout") && text.Contains("to-stderr")))
                    await Task.Delay(50);
            }
            Assert.Contains("to-stdout", text, StringComparison.Ordinal);
            Assert.Contains("to-stderr", text, StringComparison.Ordinal);
            _ = finished;
        }
        finally
        {
            if (server is { HasExited: false })
                server.Kill();
            server?.Dispose();
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
