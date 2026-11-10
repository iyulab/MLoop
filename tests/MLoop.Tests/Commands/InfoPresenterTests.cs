using DataLens.Analyzers;
using DataLens.Models;
using MLoop.CLI.Commands;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// What <c>info</c> prints about a data file. The feature ranking of <c>--analyze</c> printed the structural ranking (variance and collinearity, blind to the
/// label): where one column drove the label and another was noise, the noise column came first, the driver last, and the label
/// was listed as one of its own features.
/// </summary>
[Collection("FileSystem")]
public class InfoPresenterTests
{
    private static FeatureReport Report() => new()
    {
        TargetColumn = "y",
        Importance = new FeatureImportanceSummary
        {
            // structural: the label and a noise column rank above the real driver
            Scores = [new() { Name = "c", Score = 0.97 }, new() { Name = "b", Score = 0.79 }, new() { Name = "a", Score = 0.01 }, new() { Name = "y", Score = 0.01 }],
        },
        Permutation = new PermutationSummary
        {
            BaselineScore = 0.99,
            Features = [new() { Name = "a", Importance = 0.99 }, new() { Name = "b", Importance = 0.008 }, new() { Name = "c", Importance = 0.0001 }, new() { Name = "y", Importance = 5.0 }],
        },
    };

    private static string Render(FeatureReport report, IReadOnlyList<string>? fileColumns)
    {
        var original = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer), Ansi = AnsiSupport.No, ColorSystem = ColorSystemSupport.NoColors });
            InfoPresenter.DisplayFeatureImportance(report, "y", fileColumns);
            return buffer.ToString();
        }
        finally
        {
            AnsiConsole.Console = original;
        }
    }

    [Fact]
    public void TheDriverComesFirst_TheLabelIsNotItsOwnFeature_AndTheMethodIsNamed()
    {
        var text = Render(Report(), ["a", "b", "c", "y"]);

        Assert.Contains("Importance (permutation)", text, StringComparison.Ordinal);
        var rows = text.Split('\n').Where(l => l.StartsWith("│ ", StringComparison.Ordinal)).Select(l => l.Split('│')[1].Trim()).ToList();
        Assert.Equal(["Feature", "a", "b", "c"], rows);
    }

    [Fact]
    public void AColumnTheRankingLeftOut_IsNamed()
    {
        var text = Render(Report(), ["a", "b", "c", "grp", "y"]);

        Assert.Contains("not ranked: grp", text, StringComparison.Ordinal);
    }

    [Fact]
    public void ARankingWithNoTargetAwareSource_SaysItDoesNotAnswerWhichColumnPredictsTheLabel()
    {
        var report = new FeatureReport
        {
            TargetColumn = "y",
            Importance = new FeatureImportanceSummary { Scores = [new() { Name = "a", Score = 0.9 }, new() { Name = "b", Score = 0.8 }] },
        };

        var text = Render(report, ["a", "b", "y"]);

        Assert.Contains("Importance (structural)", text, StringComparison.Ordinal);
        Assert.Contains("not from the label", System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " "), StringComparison.Ordinal);   // the line wraps at the console width
    }

    [Fact]
    public void TheLabelStatistics_StdDevIsTheSampleStandardDeviation_AsTheColumnTableSaysIt()
    {
        // 1..30, once each: the sample standard deviation is 8.8034 (the population one 8.6554).
        var distribution = Enumerable.Range(1, 30).ToDictionary(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture), _ => 1);
        var original = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer), Ansi = AnsiSupport.No, ColorSystem = ColorSystemSupport.NoColors });
            InfoPresenter.DisplayLabelDistribution(distribution, 30);
        }
        finally
        {
            AnsiConsole.Console = original;
        }

        Assert.Matches(@"Std Dev\s*│\s*8\.8034", buffer.ToString());
    }
}
