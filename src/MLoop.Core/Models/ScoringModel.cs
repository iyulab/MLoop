using Microsoft.ML;
using MLoop.Core.Runtime;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// The one way a model is loaded to score with — predict (CSV and JSON), serve, evaluate and the
/// threshold search all go through it, so none answers differently from another.
/// </summary>
/// <remarks>
/// Loading a model to score is several things, not one: the native runtime a deep-learning task needs
/// has to be present before deserialization; the probability calibration and the decision threshold
/// beside the model (<see cref="ProbabilityCalibration"/>, <see cref="DecisionThreshold"/>) have to
/// be applied, in that order; and a regression model may carry a σ-model
/// sibling (<see cref="ExperimentLayout.ResidualModelFileName"/>) that widens its band per row. Each
/// path used to do its own subset — the threshold at five sites, the sibling's path rebuilt at four —
/// and a path that forgot one gave a different answer to the same row.
/// </remarks>
public static class ScoringModel
{
    /// <summary>
    /// The model at <paramref name="modelPath"/>, ready to score: its task's runtime ensured (when
    /// <paramref name="task"/> is given), then the <paramref name="layers"/> beside it applied in their
    /// order — the probability calibration, then the decision threshold, which answers from the
    /// calibrated probability. A caller that converts the saved model itself passes
    /// <see cref="ScoringLayers.None"/> (ONNX export: the graph carries the trained model, the manifest
    /// names the layers); fitting a calibration reads the model's own probabilities the same way.
    /// </summary>
    /// <remarks>
    /// <paramref name="inputSchema"/> is the schema the model was saved with, and training saves
    /// without one — it is <c>null</c> for every trained model. A layer that needs the model's output
    /// shape reads it from the data it is given (see the multiclass calibration), never from this.
    /// </remarks>
    /// <exception cref="InvalidOperationException">The task needs a runtime that is not installed.</exception>
    public static ITransformer Load(
        MLContext mlContext, string modelPath, out DataViewSchema inputSchema, string? task = null, ScoringLayers layers = ScoringLayers.All)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        if (!string.IsNullOrWhiteSpace(task))
            RuntimeManager.EnsureRuntimeForTask(task);
        var model = mlContext.Model.Load(modelPath, out inputSchema);
        if (layers.HasFlag(ScoringLayers.Calibration))
            model = ProbabilityCalibration.ApplyBeside(mlContext, model, modelPath);
        if (layers.HasFlag(ScoringLayers.DecisionThreshold))
            model = DecisionThreshold.ApplyBeside(mlContext, model, modelPath);
        return model;
    }

    /// <inheritdoc cref="Load(MLContext, string, out DataViewSchema, string?, ScoringLayers)"/>
    public static ITransformer Load(MLContext mlContext, string modelPath, string? task = null, ScoringLayers layers = ScoringLayers.All)
        => Load(mlContext, modelPath, out _, task, layers);

    /// <summary>Where the σ-model beside <paramref name="modelPath"/> lives.</summary>
    public static string ResidualPathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.ResidualModelFileName);

    /// <summary>The σ-model beside <paramref name="modelPath"/>, or <c>null</c> when the model has none.</summary>
    public static ITransformer? LoadResidual(MLContext mlContext, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var path = ResidualPathBeside(modelPath);
        return File.Exists(path) ? mlContext.Model.Load(path, out _) : null;
    }
}

/// <summary>What <see cref="ScoringModel.Load(MLContext, string, string?, ScoringLayers)"/> applies on top of the saved model.</summary>
[Flags]
public enum ScoringLayers
{
    /// <summary>The saved model as trained.</summary>
    None = 0,
    /// <summary>The probability calibration beside it (<see cref="ProbabilityCalibration"/>).</summary>
    Calibration = 1,
    /// <summary>The decision threshold beside it (<see cref="DecisionThreshold"/>).</summary>
    DecisionThreshold = 2,
    /// <summary>Everything a scoring path answers with.</summary>
    All = Calibration | DecisionThreshold,
}
