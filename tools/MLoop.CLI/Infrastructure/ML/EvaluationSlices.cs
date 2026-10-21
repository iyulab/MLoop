using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// An evaluation per value of one column — <c>mloop evaluate --slice-by</c> and <c>POST /evaluate</c> with
/// <c>sliceBy</c> measure each slice the same way the whole file is measured, so a slice's numbers mean what the
/// overall numbers mean.
/// </summary>
public static class EvaluationSlices
{
    /// <summary>One value of the slicing column: its rows, and its metrics or why there are none.</summary>
    public sealed record Slice(string Value, int Rows, Dictionary<string, double>? Metrics, string? Error, bool Small,
        ConfusionCounts? Confusion = null);

    /// <summary>Below this many rows a slice's metrics are reported but marked: they move a lot from one sample to the next.</summary>
    public const int SmallRows = 30;

    /// <summary>
    /// Each value of <paramref name="sliceBy"/> in <paramref name="testDataFile"/>, measured. A slice the evaluator
    /// cannot score (one class only, for AUC) reports why instead of failing the whole evaluation.
    /// </summary>
    public static async Task<IReadOnlyList<Slice>> EvaluateAsync(
        EvaluationEngine engine, string sliceBy, string testDataFile, string modelPath, ExperimentData experiment,
        string? weightColumn, CancellationToken cancellationToken = default)
    {
        var directory = Path.Combine(Path.GetTempPath(), "mloop-slices-" + Guid.NewGuid().ToString("N"));
        try
        {
            var parts = await CsvPartition.ByColumnAsync(testDataFile, sliceBy, directory);
            var results = new List<Slice>();
            foreach (var part in parts)
            {
                try
                {
                    var evaluation = await engine.EvaluateAsync(
                        modelPath, part.Path, experiment.Config.LabelColumn, experiment.Task,
                        cancellationToken, experiment.Config.InputSchema,
                        experiment.Config.GroupColumn, experiment.Config.UserColumn, experiment.Config.ItemColumn,
                        weightColumn);
                    results.Add(new Slice(part.Value, part.Rows, evaluation.Metrics, null, part.Rows < SmallRows, evaluation.Confusion));
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    results.Add(new Slice(part.Value, part.Rows, null, ex.Message, part.Rows < SmallRows));
                }
            }
            return results;
        }
        finally
        {
            try { Directory.Delete(directory, recursive: true); } catch { /* temp files; the OS reclaims them */ }
        }
    }
}
