using System.Text.RegularExpressions;
using MLoop.CLI.Commands;
using MLoop.Core.Models;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// The data summary <c>train</c> prints before it starts. Its row count was the number of lines after the header, so a file that
/// ended in blank lines said more rows than the label check beside it, which skips them, and than the loader reads.
/// </summary>
[Collection("FileSystem")]
public class TrainPresenterTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-train-presenter-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string Summary(string csv)
    {
        var path = Path.Combine(_dir, "data.csv");
        File.WriteAllText(path, csv);
        var original = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer), Ansi = AnsiSupport.No, ColorSystem = ColorSystemSupport.NoColors });
            TrainPresenter.DisplayDataSummary(path, "y");
            return buffer.ToString();
        }
        finally
        {
            AnsiConsole.Console = original;
        }
    }

    private static string Rows(string summary) => Regex.Match(summary, @"Rows\s*│\s*([\d,]+)").Groups[1].Value;

    [Fact]
    public void TheRowCount_SkipsTheBlankLinesOfATable()
    {
        Assert.Equal("3", Rows(Summary("x,y\n1,a\n2,b\n3,a\n")));
        Assert.Equal("3", Rows(Summary("x,y\n1,a\n2,b\n3,a\n\n\n")));
        Assert.Equal("3", Rows(Summary("x,y\n1,a\n\n2,b\n3,a\n")));
    }

    [Fact]
    public void TheFeatureCount_IsCalledWhatItIs_TheColumnsBesideTheLabelBeforeAnyExclusion()
    {
        var summary = Summary("x,z,y\n1,2,a\n3,4,b\n");

        Assert.Matches(@"Candidate features\s*│\s*2", summary);
        Assert.DoesNotMatch(@"│\s*Features\s*│", summary);
    }

    [Fact]
    public void TheValidatedLine_GivesTheScoreAndItsStandardError_BesideTheRowsOfTheTestTable()
    {
        var line = TrainPresenter.ValidatedLine(new ValidatedScore("r_squared", 0.4316, 0.1021, 5), 88);

        Assert.Contains("R SQUARED 0.4316 ± 0.102 (standard error over 5 folds)", line, StringComparison.Ordinal);
        Assert.Contains("one split of 88 rows", line, StringComparison.Ordinal);
    }

    [Fact]
    public void TheValidatedLine_NamesTheSpreadOfSlices_ForAHoldout_AndLeavesOutRowsItDoesNotKnow()
    {
        var line = TrainPresenter.ValidatedLine(new ValidatedScore("auc", 0.9, 0.01, 1), null);

        Assert.Contains("spread of the validation slices", line, StringComparison.Ordinal);
        Assert.DoesNotContain("one split of", line, StringComparison.Ordinal);
    }
}
