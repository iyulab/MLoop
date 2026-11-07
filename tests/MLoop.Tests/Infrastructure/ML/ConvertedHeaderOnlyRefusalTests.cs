using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using Parquet.Serialization;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// A workbook with a header row and nothing under it is refused as that, naming the workbook, as a CSV of the same shape is. The
/// refusal read only comma- and tab-separated files, so the converted table of a workbook went on: it trained as a label with no
/// values and ended in "Cannot train a classifier with only one class" - advice about the label for a file that has no rows.
/// </summary>
[Collection("FileSystem")]
public sealed class ConvertedHeaderOnlyRefusalTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-convhdr-" + Guid.NewGuid().ToString("N"));

    public ConvertedHeaderOnlyRefusalTests() => Directory.CreateDirectory(_workspace);

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string HeaderOnlyWorkbook(string name = "members.xlsx")
        => WorkbookFile.Write(Path.Combine(_workspace, name), ["x1", "x2", "y"], Enumerable.Empty<double[]>());

    private string WorkbookWithRows(string name = "rows.xlsx")
        => WorkbookFile.Write(Path.Combine(_workspace, name), ["x1", "x2", "y"], [new[] { 1.0, 2.0, 0.0 }, new[] { 2.0, 3.0, 1.0 }]);

    [Fact]
    public async Task ConvertedHeaderOnlyWorkbook_IsRefused_NamingTheWorkbook()
    {
        var workbook = HeaderOnlyWorkbook();
        var table = await TabularDataFile.AsCsvAsync(workbook);

        var refusal = TrainingInput.HeaderOnlyDataFileRefusal(table, "binary-classification", reportedAs: workbook);

        Assert.NotNull(refusal);
        Assert.Contains("members.xlsx", refusal, StringComparison.Ordinal);
        Assert.Contains("no data rows", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("mloop_table_", refusal, StringComparison.Ordinal); // the scratch copy is not the user's file
    }

    private string WorkbookWithNoCells(string name = "blank.xlsx")
        => WorkbookFile.Write(Path.Combine(_workspace, name), Array.Empty<string>(), Enumerable.Empty<double[]>());

    [Fact]
    public async Task ConvertedWorkbookWithRows_IsNotRefused()
    {
        var workbook = WorkbookWithRows();
        var table = await TabularDataFile.AsCsvAsync(workbook);

        Assert.Null(TrainingInput.HeaderOnlyDataFileRefusal(table, "binary-classification", reportedAs: workbook));
    }

    [Fact]
    public async Task ConvertingAWorkbookWithNoCells_IsRefusedAsEmpty_NamingTheWorkbook()
    {
        var error = await Assert.ThrowsAsync<EmptyTableFileException>(() => TabularDataFile.AsCsvAsync(WorkbookWithNoCells()));

        Assert.Contains("blank.xlsx is empty", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("mloop_table_", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ServerPreparation_OfAWorkbookWithNoCells_Throws_NamingTheWorkbookNotTheParser()
    {
        var definition = new ModelDefinition { Task = "binary-classification", Label = "y" };

        var error = await Assert.ThrowsAnyAsync<InvalidOperationException>(
            () => TrainingInput.PrepareAsync(WorkbookWithNoCells(), definition, maxRows: null, projectRoot: _workspace));

        Assert.Contains("blank.xlsx is empty", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("No header record", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ServerPreparation_OfAHeaderOnlyWorkbook_Throws_NamingTheWorkbookNotTheLabel()
    {
        var definition = new ModelDefinition { Task = "binary-classification", Label = "y" };

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => TrainingInput.PrepareAsync(HeaderOnlyWorkbook(), definition, maxRows: null, projectRoot: _workspace));

        Assert.Contains("members.xlsx", error.Message, StringComparison.Ordinal);
        Assert.Contains("no data rows", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("only one class", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Train_OnAHeaderOnlyWorkbook_IsRefusedAsThat_NotAsOneClass()
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var workbook = HeaderOnlyWorkbook("members.xlsx");

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train", workbook, "--label", "y", "--time", "5");   // no search: refused before it trains

        var output = stdout + stderr;
        Assert.Equal(1, exit);
        Assert.Contains("members.xlsx has a header row and no data rows", output, StringComparison.Ordinal);
        Assert.DoesNotContain("only one class", output, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Train_OnAWorkbookWithNoCells_IsRefusedAsEmpty_NotWithTheParsersState()
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train", WorkbookWithNoCells(), "--label", "y", "--time", "5");   // no search: refused before it trains

        var output = stdout + stderr;
        Assert.Equal(1, exit);
        Assert.Contains("blank.xlsx is empty", output, StringComparison.Ordinal);
        Assert.DoesNotContain("No header record", output, StringComparison.Ordinal);
        Assert.DoesNotContain("IReader state", output, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Train_OnAWorkbookWhoseLabelIsNotAColumn_NamesTheWorkbook_NotTheScratchTable()
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train", WorkbookWithRows("people.xlsx"), "--label", "nope", "--time", "5");   // no search: refused before it trains

        var output = stdout + stderr;
        Assert.Equal(1, exit);
        Assert.Contains("Label column 'nope' not found in people.xlsx", output, StringComparison.Ordinal);
        Assert.Contains("Available columns: x1, x2, y", output, StringComparison.Ordinal);
        Assert.DoesNotContain("mloop_table_", output, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("info")]
    [InlineData("analyze profile")]
    [InlineData("sample create --from")]
    [InlineData("detect")]
    public async Task EveryCommandThatReadsATable_NamesAWorkbookWithNoCells_NotTheParsersState(string command)
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        var (exit, stdout, stderr) = await CliRunner.RunAsync([.. command.Split(' '), WorkbookWithNoCells()]);

        var output = stdout + stderr;
        Assert.NotEqual(0, exit);
        Assert.Contains("blank.xlsx is empty", output, StringComparison.Ordinal);
        Assert.DoesNotContain("No header record", output, StringComparison.Ordinal);
        Assert.DoesNotContain("IReader state", output, StringComparison.Ordinal);
        Assert.DoesNotContain("mloop_table_", output, StringComparison.Ordinal);
    }

    public sealed class Measure
    {
        public double X1 { get; set; }
        public int Y { get; set; }
    }

    [Fact]
    public async Task Train_OnAParquetFileWithASchemaAndNoRows_IsRefusedAsHeaderOnly()
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var parquet = Path.Combine(_workspace, "none.parquet");
        await ParquetSerializer.SerializeAsync(Array.Empty<Measure>(), parquet);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train", parquet, "--label", "Y", "--time", "5");   // no search: refused before it trains

        var output = stdout + stderr;
        Assert.Equal(1, exit);
        Assert.Contains("none.parquet has a header row and no data rows", output, StringComparison.Ordinal);
        Assert.DoesNotContain("only one class", output, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Merging_AFileWithNoCells_IsRefusedNamingIt_NotThrownPastTheCaller()
    {
        var table = Path.Combine(_workspace, "a.csv");
        await File.WriteAllTextAsync(table, "x1,x2,y\n1,2,0\n2,3,1\n");

        var merge = await TrainingDataFiles.MergeAsync([table, WorkbookWithNoCells("b.xlsx")], weightColumn: null, Path.Combine(_workspace, "merged.csv"));

        Assert.Null(merge.Path);
        Assert.Contains("b.xlsx is empty", merge.Error, StringComparison.Ordinal);
        Assert.NotNull(merge.Guidance);
    }
}
