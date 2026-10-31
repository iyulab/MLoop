using System.Reflection;
using DataLens;
using DataLens.Models;
using MLoop.CLI.Infrastructure.Diagnostics;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Wraps DataLens analysis API with graceful degradation.
/// All methods return null when the library is unavailable or on error.
/// Replaces the former InsightAnalyzer (UInsight direct wrapper).
/// </summary>
internal sealed class DataLensAnalyzer
{
    /// <summary>Whether DataLens is available for analysis.</summary>
    public bool IsAvailable { get; }

    /// <summary>DataLens version string, or null if unavailable.</summary>
    public string? Version { get; }

    public DataLensAnalyzer()
    {
        try
        {
            // Verify DataLens can be loaded by accessing its engine type
            _ = typeof(DataLensEngine);
            IsAvailable = true;
            Version = typeof(DataLensEngine).Assembly.GetName().Version?.ToString();
        }
        catch
        {
            IsAvailable = false;
        }
    }

    /// <summary>
    /// The options for the analyses <paramref name="choose"/> turns on, and no other: every boolean switch of <see cref="AnalysisOptions"/> starts off
    /// (the <c>Include…</c> ones and any other, such as <c>ComputeSilhouette</c>, which defaults to on and does not carry the prefix).
    /// DataLens runs whatever its defaults say, and a switch added in a later version defaults to on — change points came in that way and
    /// ran on every <c>mloop info --analyze</c> and <c>mloop analyze</c>, quadratic in the rows (ten seconds for fifty thousand), though nothing here reads them.
    /// Naming what is wanted keeps the next one out until someone wants it.
    /// </summary>
    public static AnalysisOptions Only(Action<AnalysisOptions> choose)
    {
        ArgumentNullException.ThrowIfNull(choose);
        var options = new AnalysisOptions();
        foreach (var property in typeof(AnalysisOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            if (property.PropertyType == typeof(bool) && property.CanWrite)
                property.SetValue(options, false);
        choose(options);
        return options;
    }

    /// <summary>
    /// Profiles a CSV file and returns column summaries.
    /// Returns null if unavailable or on error.
    /// </summary>
    public async Task<ProfileReport?> ProfileAsync(string csvPath)
    {
        if (!IsAvailable) return null;
        try
        {
            return await DataLensEngine.Profile(csvPath);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[grey]DataLens profile failed: {ex.GetType().Name}: {Markup.Escape(ex.Message)}[/]");
            return null;
        }
    }

    /// <summary>
    /// Runs full analysis on a CSV file with specified options.
    /// Returns null if unavailable or on error.
    /// </summary>
    public async Task<AnalysisResult?> AnalyzeAsync(string csvPath, AnalysisOptions? options = null)
    {
        if (!IsAvailable) return null;
        try
        {
            var result = await DataLensEngine.Analyze(csvPath, options);
            foreach (var warning in result.Warnings)
                if (AnalysisLimit(warning) is { } limit)
                    WarningConsole.Warn(limit);
            return result;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]DataLens analysis failed: {ex.GetType().Name}: {Markup.Escape(ex.Message)}[/]");
            return null;
        }
    }

    /// <summary>
    /// What a DataLens warning means for the answer, when it means the analysis did not run or ran on less than the file holds - the cases where
    /// an empty or short result would otherwise read as a finding (<c>analyze correlation</c> on two rows said "no high-correlation pairs"). Warnings
    /// that describe the data instead (duplicate rows, a column pair that is the same column, a covariance that may be singular) are not passed on:
    /// they are not about whether the analysis ran, and the commands show the pairs themselves. Null for those.
    /// </summary>
    internal static string? AnalysisLimit(AnalysisWarning warning)
    {
        var name = warning.Analyzer;
        var columns = warning.AffectedColumns is { Count: > 0 } affected ? string.Join(", ", affected) : null;
        return warning.Category switch
        {
            WarningCategory.InsufficientRows => $"The {name} analysis did not run (too few rows); an empty result here says nothing about the data.",
            WarningCategory.InsufficientColumns => $"The {name} analysis did not run (too few numeric columns); an empty result here says nothing about the data.",
            WarningCategory.InsufficientUsableColumns => $"The {name} analysis did not run (too few columns are left once constant and empty ones are set aside); an empty result here says nothing about the data.",
            WarningCategory.DegenerateColumnsExcluded => $"The {name} analysis left out {(columns is null ? "constant or empty columns" : $"the constant or empty column(s) {columns}")}.",
            WarningCategory.AllPairsUndefined => $"The {name} analysis could not compute a value for any pair of columns; an empty result here says nothing about the data.",
            WarningCategory.ComputationFailed or WarningCategory.UpstreamError => $"The {name} analysis failed and is not in the result: {warning.Message}",
            _ => null,
        };
    }
}
