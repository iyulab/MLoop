namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop train --data a --data b</c> merges files of any format it can read, not only CSV.
/// </summary>
/// <remarks>
/// The merge read every file as CSV text. A JSON file's whole content became its "header", so two
/// JSON files of the same shape were refused as having no column in common — the error listed the
/// raw JSON as each file's columns. Each file is now read as a table first, as a single data file is.
/// <c>--analyze-data</c> stops after the data is prepared, so no model is trained here.
/// </remarks>
public class TrainMultiFileMergeTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-merge-" + Guid.NewGuid().ToString("N"));

    public TrainMultiFileMergeTests()
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
    public async Task Json_files_are_merged_as_tables()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "regression");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/a.json", """[{"x":1,"y":2},{"x":2,"y":4}]""");
        await File.WriteAllTextAsync("datasets/b.json", """[{"x":3,"y":6},{"x":4,"y":8},{"x":5,"y":10}]""");

        var (exit, coloured, _) = await CliRunner.RunAsync(
            "train", "--data", "datasets/a.json", "--data", "datasets/b.json", "--label", "y", "--analyze-data");
        // The words, not the colours the console puts around the counts.
        var output = System.Text.RegularExpressions.Regex.Replace(coloured, @"\e\[[0-9;]*m", "");

        Assert.True(exit == 0, output);
        Assert.Contains("Merged 5 rows from 2 files", output);
        Assert.Contains("a.json: 2 rows", output);
        var merged = await File.ReadAllLinesAsync("datasets/merged_train.csv");
        Assert.Equal(["x", "y"], merged[0].TrimStart('﻿').Split(','));
        Assert.Equal(6, merged.Length);
    }
}
