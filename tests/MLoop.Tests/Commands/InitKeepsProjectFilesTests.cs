using MLoop.CLI;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop init</c> scaffolds what is missing and leaves a project's own files alone.
/// </summary>
/// <remarks>
/// A cloned repository carries its <c>mloop.yaml</c>, README and <c>.gitignore</c> but not the
/// <c>.mloop/</c> marker, so <c>init .</c> is the first command run in it — and it used to replace all
/// three with the template, the project's models, tasks and labels included, without saying so.
/// </remarks>
public class InitKeepsProjectFilesTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-init-keeps-" + Guid.NewGuid().ToString("N"));

    private const string ProjectYaml = """
        project: iris
        models:
          default:
            task: multiclass-classification
            label: Species
        """;

    public InitKeepsProjectFilesTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { /* nothing left to do */ } }

        if (Directory.Exists(_workspace))
        {
            try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
        }
    }

    [Fact]
    public async Task InitInAClonedProject_KeepsItsConfigurationReadmeAndGitignore()
    {
        File.WriteAllText("mloop.yaml", ProjectYaml);
        File.WriteAllText("README.md", "# A tutorial");
        File.WriteAllText(".gitignore", "secrets/\n");

        var (exit, output, _) = await CliRunner.RunAsync("init", ".", "--task", "binary-classification", "--label", "Churn");

        Assert.True(exit == 0, $"init exited {exit}: {output}");
        Assert.Equal(ProjectYaml, File.ReadAllText("mloop.yaml"));
        Assert.Equal("# A tutorial", File.ReadAllText("README.md"));
        Assert.Equal("secrets/\n", File.ReadAllText(".gitignore"));
        Assert.True(Directory.Exists(".mloop"), "the project marker is what init exists to create");
        Assert.Contains("mloop.yaml", output);
        Assert.Contains("not applied", output);
    }

    [Fact]
    public async Task ReinitializingWithForce_RebuildsTheMarker_AndStillKeepsTheConfiguration()
    {
        var (first, firstOutput, _) = await CliRunner.RunAsync("init", ".", "--task", "regression", "--label", "Price");
        Assert.True(first == 0, $"init exited {first}: {firstOutput}");
        File.WriteAllText("mloop.yaml", ProjectYaml);

        var (exit, output, _) = await CliRunner.RunAsync("init", ".", "--force");

        Assert.True(exit == 0, $"init --force exited {exit}: {output}");
        Assert.Equal(ProjectYaml, File.ReadAllText("mloop.yaml"));
    }

    [Fact]
    public async Task InitInAnEmptyFolder_WritesTheTemplate()
    {
        var (exit, output, _) = await CliRunner.RunAsync("init", ".", "--task", "regression", "--label", "Price");

        Assert.True(exit == 0, $"init exited {exit}: {output}");
        Assert.Contains("label: Price", File.ReadAllText("mloop.yaml"));
        Assert.DoesNotContain("Kept what was already there", output);
    }
}
