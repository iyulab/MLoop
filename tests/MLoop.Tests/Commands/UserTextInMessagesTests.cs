namespace MLoop.Tests.Commands;

/// <summary>
/// A path or name the user typed is shown in an error as typed. Error and warning lines are Spectre markup, so a path
/// with brackets (<c>data[1].csv</c>) read as a style tag: the command printed "Unbalanced markup stack" instead of
/// the path it could not find, or failed with an unhandled exception.
/// </summary>
[Collection("FileSystem")]
public sealed class UserTextInMessagesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-usertext-{Guid.NewGuid():N}");
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public UserTextInMessagesTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        File.WriteAllText(Path.Combine(_root, "mloop.yaml"), "project: p\nmodels:\n  default:\n    task: binary-classification\n    label: y\n");
        Directory.SetCurrentDirectory(_root);
    }

    public void Dispose()
    {
        Directory.SetCurrentDirectory(_originalDirectory);
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Theory]
    [InlineData("sample", "create", "--from", "rows[1].csv")]
    [InlineData("train", "rows[red].csv", "--label", "y")]
    [InlineData("train", "--data", "rows[1].csv", "--label", "y")]
    public async Task AMissingFileWithBrackets_IsNamedAsTyped(params string[] args)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync(args);

        Assert.NotEqual(0, exit);
        var output = stdout + stderr;
        Assert.Contains("rows[", output);
        Assert.DoesNotContain("Unbalanced markup", output);
        Assert.DoesNotContain("Unhandled", output);
    }

    // The file named on the command line is the one reported missing — not "no data file specified".
    [Fact]
    public async Task TrainingAMissingFile_SaysThatFileIsNotFound()
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync("train", "rows[1].csv", "--label", "y");

        Assert.Equal(1, exit);
        Assert.Contains("Data file not found: rows[1].csv", stdout + stderr);
        Assert.DoesNotContain("No data file specified", stdout + stderr);
    }

    // Progress lines name the data's own columns: a column called score[1] is a column, not a style tag.
    [Fact]
    public async Task AColumnNamedWithBrackets_IsShownByInfo()
    {
        File.WriteAllText("rows.csv", "x,score[1]\n1,a\n2,b\n3,a\n4,b\n");

        var (exit, stdout, stderr) = await CliRunner.RunAsync("info", "rows.csv", "--label", "score[1]");

        Assert.True(exit == 0, stdout + stderr);
        Assert.Contains("score[1]", stdout + stderr);
        Assert.DoesNotContain("Unbalanced markup", stdout + stderr);
    }

    [Fact]
    public async Task InstallingAnEncoderNotInTheCatalog_PrintsTheAddCommandWithItsOptionalTokenizer()
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync("encoder", "install", "x[1]");

        Assert.Equal(1, exit);
        Assert.Contains("'x[1]' is not in the encoder catalog", stdout + stderr);
        Assert.Contains("[--tokenizer <tokenizer.model>]", stdout + stderr);
    }
}
