using System.CommandLine;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>info --json</c> reports the same profiling data <c>info</c>'s human report shows —
/// column classification, the DataLens profile, and (when <c>--analyze</c> is passed) the deep
/// analysis result serialized as-is. Exercises the real command tree
/// (<see cref="Program.BuildRootCommand"/>), same approach as the other <c>--json</c> command tests.
/// Also the first test in this repo to run <c>info</c> through the real Spectre-rendering path with
/// <c>--analyze</c> — that run is what surfaced two serialization gaps in this new code (NaN/Infinity
/// doubles, and DataLens's 2D-array report fields) that a string-level unit test never would have.
/// </summary>
[Collection("FileSystem")]
public class InfoCommandJsonTests : IDisposable
{
    private readonly string _testDir;
    private readonly string _originalDirectory;

    public InfoCommandJsonTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-info-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
        Directory.SetCurrentDirectory(_testDir);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }

        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    private string WriteDataCsv() =>
        WriteCsv("data.csv", "a,b,Label\n1,2.5,yes\n2,,no\n3,4.5,yes\n4,5.5,no\n5,,yes\n");

    private string WriteCsv(string name, string content)
    {
        var path = Path.Combine(_testDir, name);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public async Task Info_Json_FileNotFound_ReportsError()
    {
        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", "nope.csv", "--json");

        Assert.Equal(1, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Contains(errors, e => e.GetString()!.Contains("not found"));
    }

    [Fact]
    public async Task Info_Json_EmptyFile_ReportsErrorAndNonZeroExit()
    {
        var path = WriteCsv("empty.csv", "");

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", path, "--json");

        Assert.Equal(1, exitCode); // sibling to the file-not-found case above, not a silent success
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Contains(errors, e => e.GetString()!.Contains("empty"));
    }

    [Fact]
    public async Task Info_Json_WithoutALabel_DoesNotCallAnyColumnTheLabel()
    {
        // Outside a project and without --label, nothing names the label. The column guessed for
        // ML.NET's type inference (the last one, here a start position) was reported as "Label" —
        // a role the user never gave it.
        var path = WriteCsv("qa.csv", "context,question,answer,answer_start\nseoul is big,what,big,9\nbusan is far,where,far,7\n");

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", path, "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var purposes = doc.RootElement.GetProperty("columns").EnumerateArray()
            .ToDictionary(c => c.GetProperty("name").GetString()!, c => c.GetProperty("purpose").GetString());
        Assert.DoesNotContain("Label", purposes.Values);
        Assert.Equal("Numeric Feature", purposes["answer_start"]);
    }

    [Fact]
    public async Task Info_Json_Profile_ReportsColumnsAndLabelDistribution()
    {
        var path = WriteDataCsv();

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", path, "--label", "Label", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.False(doc.RootElement.GetProperty("analyze").GetBoolean());
        Assert.False(doc.RootElement.TryGetProperty("analysis", out _)); // null omitted, --analyze not passed

        var columns = doc.RootElement.GetProperty("columns").EnumerateArray().ToList();
        Assert.Equal(3, columns.Count);
        var labelColumn = columns.Single(c => c.GetProperty("name").GetString() == "Label");
        Assert.Equal("Label", labelColumn.GetProperty("purpose").GetString());

        var labelDistribution = doc.RootElement.GetProperty("labelDistribution");
        Assert.Equal(3, labelDistribution.GetProperty("yes").GetInt32());
        Assert.Equal(2, labelDistribution.GetProperty("no").GetInt32());
    }

    [Fact]
    public async Task Info_Json_Analyze_ReportsDeepAnalysisResult()
    {
        var path = WriteDataCsv();

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", path, "--label", "Label", "--analyze", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.True(doc.RootElement.GetProperty("analyze").GetBoolean());

        var analysis = doc.RootElement.GetProperty("analysis");
        // Regression guard: DataLens's CorrelationReport.Matrix is a double[,], which
        // System.Text.Json has no built-in support for — this is what verifies the custom
        // converter actually runs end-to-end rather than only in isolation.
        var matrix = analysis.GetProperty("correlation").GetProperty("matrix");
        Assert.Equal(2, matrix.GetArrayLength());
        Assert.Equal(2, matrix[0].GetArrayLength());
    }

    [Fact]
    public async Task Info_Human_Unaffected_ByJsonChanges()
    {
        var path = WriteDataCsv();

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", path, "--label", "Label");

        Assert.Equal(0, exitCode);
        Assert.Contains("Column Information", stdout);
        Assert.Contains("Label", stdout);
        Assert.DoesNotContain("{", stdout); // no JSON leaked into the human report
    }

    private string WriteImageFolder(params (string Class, int Images)[] classes)
    {
        var root = Path.Combine(_testDir, "images");
        foreach (var (cls, images) in classes)
        {
            Directory.CreateDirectory(Path.Combine(root, cls));
            for (int i = 0; i < images; i++)
                File.WriteAllBytes(Path.Combine(root, cls, $"img{i}.png"), [0x89, 0x50, 0x4E, 0x47]);
        }
        return root;
    }

    [Fact]
    public async Task Info_ImageFolder_ReportsImagesPerClass()
    {
        // An existing absolute folder was reported as "File not found", with a tip blaming a
        // relative path — the one thing it wasn't.
        var root = WriteImageFolder(("ripe", 3), ("unripe", 2));

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", root, "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var dist = doc.RootElement.GetProperty("labelDistribution");
        Assert.Equal(3, dist.GetProperty("ripe").GetInt32());
        Assert.Equal(2, dist.GetProperty("unripe").GetInt32());
        Assert.False(doc.RootElement.TryGetProperty("errors", out _));
    }

    [Fact]
    public async Task Info_ImageFolder_HumanReport_NamesTheFolderLayout()
    {
        var root = WriteImageFolder(("ripe", 3), ("unripe", 2));

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", root);

        Assert.Equal(0, exitCode);
        Assert.Contains("unripe", stdout);
        Assert.Contains("5 images", stdout);
        Assert.DoesNotContain("not found", stdout, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("--balance", stdout); // a tabular-training remedy
    }

    [Fact]
    public async Task Info_FolderWithoutImageClasses_SaysItIsAFolder()
    {
        var dir = Path.Combine(_testDir, "tables");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "a.csv"), "x,y\n1,2\n");

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", dir, "--json");

        Assert.Equal(1, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().Select(e => e.GetString()!).ToList();
        Assert.Contains(errors, e => e.Contains("is a folder"));
        Assert.DoesNotContain(errors, e => e.Contains("not found"));
    }

    [Fact]
    public async Task Info_Json_ReportsAColumnTrainingDrops_AsExcluded_WithTheReason()
    {
        // A list column (how a Parquet list arrives) was reported as a "Text Feature"; training drops
        // it, and info now says so, from the same decision training uses.
        var rows = Enumerable.Range(0, 30).Select(i => $"{i},\"[{i},{i + 1}]\",{(i % 2 == 0 ? "yes" : "no")}");
        var path = WriteCsv("lists.csv", string.Join(Environment.NewLine, rows.Prepend("x,scores,Label")) + Environment.NewLine);

        var (exitCode, stdout, _) = await CliRunner.RunAsync("info", path, "--label", "Label", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var scores = doc.RootElement.GetProperty("columns").EnumerateArray()
            .Single(c => c.GetProperty("name").GetString() == "scores");
        Assert.Equal("Excluded (Structured)", scores.GetProperty("purpose").GetString());
    }
}
