using DataLens;
using DataLens.Models;
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
            return await DataLensEngine.Analyze(csvPath, options);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]DataLens analysis failed: {ex.GetType().Name}: {Markup.Escape(ex.Message)}[/]");
            return null;
        }
    }
}
