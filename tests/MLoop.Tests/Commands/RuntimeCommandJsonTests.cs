using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// BD-7: <c>runtime list</c> already collects <c>RuntimeStatus</c> per runtime before rendering
/// (no project context needed — it reads <c>RuntimeRegistry.All</c> directly), so a
/// <c>--json</c> branch serializes the same statuses instead of re-deriving them. Exercises the
/// real command tree (<see cref="Program.BuildRootCommand"/>), same approach as
/// <c>ValidateCommandJsonTests</c>/<c>StatusCommandJsonTests</c>.
/// </summary>
[Collection("FileSystem")]
public class RuntimeCommandJsonTests
{
    // Same AnsiConsole.Console rebinding as the other --json tests — Console.SetOut alone
    // doesn't reach Spectre's ambient renderer (cycle-199/202's discovery).
    private static async Task<(int ExitCode, string Stdout)> RunAsync(params string[] args)
    {
        var originalOut = Console.Out;
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Out = new AnsiConsoleOutput(buffer)
            });

            var result = Program.BuildRootCommand().Parse(args);
            var exitCode = await result.InvokeAsync();
            return (exitCode, buffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }

    [Fact]
    public async Task Runtime_List_Json_ListsBothKnownRuntimes()
    {
        var (exitCode, stdout) = await RunAsync("runtime", "list", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var runtimes = doc.RootElement.GetProperty("runtimes").EnumerateArray().ToList();
        Assert.Equal(2, runtimes.Count);
        Assert.Contains(runtimes, r => r.GetProperty("id").GetString() == "tf");
        Assert.Contains(runtimes, r => r.GetProperty("id").GetString() == "torch");
    }

    [Fact]
    public async Task Runtime_List_Json_NotInstalled_OmitsSizeBytes()
    {
        // This machine's install state varies, so assert on the invariant rather than a fixed
        // fixture: whichever runtimes report installed=false must carry a null sizeBytes (the
        // actual on-disk size is unknown until installed) alongside their approximateSizeMB estimate.
        var (exitCode, stdout) = await RunAsync("runtime", "list", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        foreach (var runtime in doc.RootElement.GetProperty("runtimes").EnumerateArray())
        {
            var installed = runtime.GetProperty("installed").GetBoolean();
            var sizeBytesProp = runtime.GetProperty("sizeBytes");
            if (installed)
                Assert.Equal(System.Text.Json.JsonValueKind.Number, sizeBytesProp.ValueKind);
            else
                Assert.Equal(System.Text.Json.JsonValueKind.Null, sizeBytesProp.ValueKind);

            Assert.True(runtime.GetProperty("approximateSizeMB").GetInt32() > 0);
        }
    }

    [Fact]
    public async Task Runtime_List_Human_StillRendersTable()
    {
        // Guards against the refactor (collect-then-render) silently changing the human path.
        var (exitCode, stdout) = await RunAsync("runtime", "list");

        Assert.Equal(0, exitCode);
        Assert.Contains("TensorFlow CPU", stdout);
        Assert.Contains("libtorch CPU", stdout);
    }
}
