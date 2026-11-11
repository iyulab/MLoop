using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;
using MLoop.Search.Validation;

namespace MLoop.Core.Training;

/// <summary>
/// A multiclass trial's validated part, scored as the run will serve it: its class probabilities tempered by a
/// temperature fitted to that part's own answers (<see cref="ProbabilityCalibration.FitTemperature"/> — the one
/// temperature fit; training fits the same on the kept trial's out-of-fold answers).
/// </summary>
/// <remarks>
/// The search ranks trials by what these rows say, and the run serves the winner calibrated. Ranked by the
/// probabilities a trainer gives raw, a one-versus-all tree — whose probabilities are normalized fixed sigmoids, top-label
/// ECE measured 0.3 to 0.4 — lost on log loss to a linear model on rows where it was the far better classifier, and the
/// run served the worse one (<c>FoldCalibrationTests</c> builds that pair). A temperature scales every class's log-probability alike, so the answers do not
/// move (accuracy, macro F1) and the log loss, its reduction and the ECE become those of the calibrated model.
/// One parameter fitted on the rows it is then scored on is a small optimism, the same for every trial.
/// </remarks>
internal static class FoldCalibration
{
    // The rows a temperature is fitted on: one parameter needs few, and every trial of every fold pays for the fit.
    private const int FitRows = 5000;

    /// <summary>
    /// <paramref name="scored"/> with every row's class probabilities tempered, or <paramref name="scored"/> itself where
    /// there is nothing to temper (not a multiclass part, no rows, or a model already at temperature 1).
    /// </summary>
    internal static IDataView Temper(MLContext mlContext, IDataView scored, string labelColumn, string? weightColumn)
    {
        if (OutOfFoldScores.Read(scored, labelColumn, weightColumn) is not { IsMulticlass: true, Count: > 0 } rows)
            return scored;

        // Every k-th row: a part sorted by its label still gives each class its share.
        var step = Math.Max(1, rows.Count / FitRows);
        var sample = new List<(float[] Probabilities, int Label, double Weight)>(Math.Min(rows.Count, FitRows + 1));
        for (var i = 0; i < rows.Count; i += step)
            sample.Add((rows.ClassProbabilities[i], rows.Class[i], rows.Weight[i]));

        var calibration = ProbabilityCalibration.FitTemperature(sample, "validated part", weightColumn, DateTimeOffset.UtcNow);
        if (calibration?.Temperature is not { } temperature || temperature == 1.0)
            return scored;
        return calibration.Apply(mlContext, new TransformerChain<ITransformer>()).Transform(scored);
    }
}
