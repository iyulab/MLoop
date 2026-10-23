using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>runtime list</c> already collects <c>RuntimeStatus</c> per runtime before rendering
/// (no project context needed — it reads <c>RuntimeRegistry.All</c> directly), so a
/// <c>--json</c> branch serializes the same statuses instead of re-deriving them. Exercises the
/// real command tree (<see cref="Program.BuildRootCommand"/>), same approach as
/// <c>ValidateCommandJsonTests</c>/<c>StatusCommandJsonTests</c>.
/// </summary>
[Collection("FileSystem")]
public class RuntimeCommandJsonTests
{
    [Fact]
    public async Task Runtime_List_Json_ListsEveryKnownRuntime()
    {
        var (exitCode, stdout, _) = await CliRunner.RunAsync("runtime", "list", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var ids = doc.RootElement.GetProperty("runtimes").EnumerateArray()
            .Select(r => r.GetProperty("id").GetString())
            .ToList();
        Assert.Equal(MLoop.Core.Runtime.RuntimeRegistry.All.Select(r => r.Id), ids);
        Assert.Equal(["tf", "torch", "onnxruntime"], ids);
    }

    [Fact]
    public async Task Runtime_List_Json_NotInstalled_OmitsSizeBytes()
    {
        // This machine's install state varies, so assert on the invariant rather than a fixed
        // fixture: whichever runtimes report installed=false must carry a null sizeBytes (the
        // actual on-disk size is unknown until installed) alongside their approximateSizeMB estimate.
        var (exitCode, stdout, _) = await CliRunner.RunAsync("runtime", "list", "--json");

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

    // Whether a runtime can be installed on this machine is the registry's answer for this platform — said in the list,
    // so a runtime whose package has nothing for this platform does not read as one install away.
    [Fact]
    public async Task Runtime_List_Json_SaysWhichRuntimesCanBeInstalledHere()
    {
        var (exitCode, stdout, _) = await CliRunner.RunAsync("runtime", "list", "--json");

        Assert.Equal(0, exitCode);
        var rid = MLoop.Core.Runtime.PlatformDetector.GetRuntimeIdentifier();
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        foreach (var runtime in doc.RootElement.GetProperty("runtimes").EnumerateArray())
        {
            var definition = MLoop.Core.Runtime.RuntimeRegistry.GetById(runtime.GetProperty("id").GetString()!)!;
            Assert.Equal(definition.NuGetPackages.ContainsKey(rid), runtime.GetProperty("available").GetBoolean());
        }
    }

    [Fact]
    public async Task Runtime_List_Human_StillRendersTable()
    {
        // Guards against the refactor (collect-then-render) silently changing the human path.
        var (exitCode, stdout, _) = await CliRunner.RunAsync("runtime", "list");

        Assert.Equal(0, exitCode);
        Assert.Contains("TensorFlow CPU", stdout);
        Assert.Contains("libtorch CPU", stdout);
    }
}
