using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop validate</c> compiles the project's scripts: training only warns about a script that does not
/// compile and goes on without it, so validation is where it is found before a run.
/// </summary>
public class ValidateScriptsTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-validate-scripts-{Guid.NewGuid():N}");

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private void Write(string relative, string content)
    {
        var path = Path.Combine(_root, ".mloop", "scripts", relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    private async Task<(List<ValidateCommand.ValidationError> Errors, List<ValidateCommand.ValidationWarning> Warnings)> ValidateAsync()
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();
        await ValidateCommand.ValidateScriptsAsync(_root, errors, warnings);
        return (errors, warnings);
    }

    [Fact]
    public async Task AScriptThatDoesNotCompile_IsAnError_WithTheCompilersLineCountedFromOne()
    {
        Write("metrics/broken.cs", "public class Broken { int x = ; }");

        var (errors, _) = await ValidateAsync();

        var error = Assert.Single(errors);
        Assert.Equal(".mloop/scripts/metrics/broken.cs", error.Path);
        Assert.Contains("line 1, column 31", error.Message);
        Assert.DoesNotContain(_root, error.Message);
    }

    [Fact]
    public async Task AScriptWithoutTheContract_IsAnError_AndAStrayHook_AWarning()
    {
        Write("hooks/pre-train/nothing.cs", "public class NotAHook { }");
        Write("hooks/stray.cs", "// in no stage");

        var (errors, warnings) = await ValidateAsync();

        Assert.Contains("IMLoopHook", Assert.Single(errors).Message);
        Assert.Equal(".mloop/scripts/hooks/stray.cs", Assert.Single(warnings).Path);
    }

    [Fact]
    public async Task ScriptsThatCompile_AndNoScripts_AreClean()
    {
        Assert.Empty((await ValidateAsync()).Errors);

        Write("hooks/post-train/ok.cs", """
            using System.Threading.Tasks;
            using MLoop.Extensibility.Hooks;
            public class OkHook : IMLoopHook
            {
                public string Name => "ok";
                public Task<HookResult> ExecuteAsync(HookContext ctx) => Task.FromResult(HookResult.Continue());
            }
            """);

        var (errors, warnings) = await ValidateAsync();
        Assert.Empty(errors);
        Assert.Empty(warnings);
    }
}
