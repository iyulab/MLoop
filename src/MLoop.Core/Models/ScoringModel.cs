using Microsoft.ML;
using MLoop.Core.Runtime;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// The one way a model is loaded to score with — predict (CSV and JSON), serve, evaluate and the
/// threshold search all go through it, so none answers differently from another.
/// </summary>
/// <remarks>
/// Loading a model to score is three things, not one: the native runtime a deep-learning task needs
/// has to be present before deserialization; the decision threshold chosen beside the model
/// (<see cref="DecisionThreshold"/>) has to be applied; and a regression model may carry a σ-model
/// sibling (<see cref="ExperimentLayout.ResidualModelFileName"/>) that widens its band per row. Each
/// path used to do its own subset — the threshold at five sites, the sibling's path rebuilt at four —
/// and a path that forgot one gave a different answer to the same row.
/// </remarks>
public static class ScoringModel
{
    /// <summary>
    /// The model at <paramref name="modelPath"/>, ready to score: its task's runtime ensured (when
    /// <paramref name="task"/> is given) and its decision threshold applied, unless
    /// <paramref name="withDecisionThreshold"/> is false — for a caller that converts the saved model
    /// itself (ONNX export: the graph carries the trained model, the manifest names the threshold).
    /// </summary>
    /// <exception cref="InvalidOperationException">The task needs a runtime that is not installed.</exception>
    public static ITransformer Load(
        MLContext mlContext, string modelPath, out DataViewSchema inputSchema, string? task = null, bool withDecisionThreshold = true)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        if (!string.IsNullOrWhiteSpace(task))
            RuntimeManager.EnsureRuntimeForTask(task);
        var model = mlContext.Model.Load(modelPath, out inputSchema);
        return withDecisionThreshold ? DecisionThreshold.ApplyBeside(mlContext, model, modelPath) : model;
    }

    /// <inheritdoc cref="Load(MLContext, string, out DataViewSchema, string?, bool)"/>
    public static ITransformer Load(MLContext mlContext, string modelPath, string? task = null, bool withDecisionThreshold = true)
        => Load(mlContext, modelPath, out _, task, withDecisionThreshold);

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
