using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Commands;

/// <summary>
/// An <c>--output</c> that is the data file the command reads. <c>mloop predict data.csv -o data.csv</c> opened the output for
/// writing first, which emptied the input before it was read, and failed with a stream error and a header-only file left behind;
/// <c>sample create</c> and <c>detect</c> wrote the sample or the per-point result over the data. Each says so and stops before writing.
/// </summary>
[Collection("FileSystem")]
public class OutputFileGuardTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public OutputFileGuardTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-output-guard-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private string Write(string name, params string[] lines)
    {
        var path = Path.Combine(_tempDir, name);
        File.WriteAllLines(path, lines);
        return path;
    }

    [Fact]
    public void The_same_file_by_another_spelling_is_refused_and_another_file_is_not()
    {
        var data = Write("data.csv", "a,b", "1,2");
        var other = Path.Combine(_tempDir, "other.csv");

        Assert.NotNull(OutputFileGuard.Refusal(data, data, null));
        Assert.NotNull(OutputFileGuard.Refusal(data, Path.Combine(_tempDir, ".", "sub", "..", "data.csv"), null));
        Assert.NotNull(OutputFileGuard.Refusal(data, "data.csv", _tempDir));                  // relative to the root the command writes under
        Assert.Null(OutputFileGuard.Refusal(data, other, null));
        Assert.Null(OutputFileGuard.Refusal(data, null, null));
        Assert.Null(OutputFileGuard.Refusal(null, data, null));
        Assert.Null(OutputFileGuard.Refusal(Path.Combine(_tempDir, "missing.csv"), Path.Combine(_tempDir, "missing.csv"), null));   // nothing to replace
    }

    [Fact]
    public void A_link_to_the_input_is_the_input()
    {
        if (OperatingSystem.IsWindows())
            return;   // creating a link needs a privilege there; the comparison itself is the same call
        var data = Write("data.csv", "a,b", "1,2");
        var link = Path.Combine(_tempDir, "link.csv");
        File.CreateSymbolicLink(link, data);

        Assert.NotNull(OutputFileGuard.Refusal(data, link, null));
    }

    [Fact]
    public async Task Sample_create_and_detect_leave_the_data_as_it_is()
    {
        Directory.SetCurrentDirectory(_tempDir);
        var rows = Enumerable.Range(0, 40).Select(i => $"{i},{i % 3}").Prepend("v,w").ToArray();
        var data = Write("data.csv", rows);
        var before = await File.ReadAllBytesAsync(data);

        var sample = await CliRunner.RunAsync("sample", "create", "--from", "data.csv", "--size", "5", "-o", "data.csv");
        var detect = await CliRunner.RunAsync("detect", "data.csv", "-c", "v", "-o", "data.csv");

        Assert.NotEqual(0, sample.ExitCode);
        Assert.NotEqual(0, detect.ExitCode);
        Assert.Contains("--output is the input file", sample.Stdout + sample.Stderr, StringComparison.Ordinal);
        Assert.Contains("--output is the input file", detect.Stdout + detect.Stderr, StringComparison.Ordinal);
        Assert.Equal(before, await File.ReadAllBytesAsync(data));
    }

    [Fact]
    public async Task Predict_leaves_the_data_as_it_is()
    {
        Directory.SetCurrentDirectory(_tempDir);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_tempDir, "p"));
        var data = Write("p/datasets/new.csv", "a,b", "1,2", "3,4");
        var before = await File.ReadAllBytesAsync(data);

        var result = await CliRunner.RunAsync("predict", "datasets/new.csv", "-o", "datasets/new.csv");

        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("--output is the input file", result.Stdout + result.Stderr, StringComparison.Ordinal);
        Assert.Equal(before, await File.ReadAllBytesAsync(data));
    }
}
