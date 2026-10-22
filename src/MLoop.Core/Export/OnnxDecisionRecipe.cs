using System.Globalization;
using MLoop.Core.Models;

namespace MLoop.Core.Export;

/// <summary>
/// How a consumer of an exported graph turns its outputs into the answer <c>mloop predict</c> gives:
/// the graph carries the trained model, and the layers beside it — the probability calibration, then
/// the decision threshold (<see cref="ScoringLayers"/>) — are the consumer's to apply.
/// </summary>
/// <remarks>
/// The manifest's sentence (<see cref="Describe"/>) and the arithmetic <c>--verify</c> replays
/// (<see cref="AnswerBinary"/>, <see cref="AnswerMulticlass"/>) come from this one place, so the
/// sentence is checked against what the model actually answers instead of being read by no one. It
/// was once written separately and left the calibration out: a consumer following it answered
/// differently from <c>mloop predict</c> on every calibrated model.
/// </remarks>
public sealed record OnnxDecisionRecipe(string Task, ProbabilityCalibration? Calibration, DecisionThreshold? Threshold)
{
    /// <summary>The recipe for the model at <paramref name="modelPath"/>: the layers beside it.</summary>
    public static OnnxDecisionRecipe Beside(string task, string modelPath)
        => new(task, ProbabilityCalibration.ReadBeside(modelPath), DecisionThreshold.ReadBeside(modelPath));

    private bool IsBinary => Task == "binary-classification";
    private bool IsMulticlass => Task == "multiclass-classification";

    /// <summary>Whether answering as <c>mloop predict</c> does takes more than reading the graph.</summary>
    public bool HasLayers => (IsBinary && (Calibration is { IsMulticlass: false } || Threshold is not null))
                             || (IsMulticlass && Calibration is { IsMulticlass: true });

    /// <summary>
    /// A binary answer from the graph's <paramref name="probability"/> and <paramref name="label"/>:
    /// the calibrated probability, then the label at the threshold — or, with a calibration and no
    /// threshold, at 0.5 on the calibrated probability, as the calibration step answers.
    /// </summary>
    public (float Probability, bool Label) AnswerBinary(float probability, bool label)
    {
        if (Calibration is { IsMulticlass: false } calibration)
        {
            probability = (float)calibration.Map(probability);
            label = probability >= 0.5f;
        }
        if (Threshold is { } threshold)
            label = probability >= (float)threshold.Threshold;
        return (probability, label);
    }

    /// <summary>The class probabilities from the graph's <paramref name="score"/>, tempered when calibrated.</summary>
    public float[] AnswerMulticlass(float[] score)
        => Calibration is { IsMulticlass: true } calibration ? calibration.Temper(score) : score;

    /// <summary>The rule in words, for the manifest's <c>label.decision</c>.</summary>
    /// <param name="multiclassPredicted">Multiclass: what the graph's <c>PredictedLabel</c> holds.</param>
    public string Describe(string? multiclassPredicted = null)
    {
        if (IsBinary)
            return DescribeBinary();
        if (IsMulticlass)
        {
            var reads = $"Score holds one probability per class, in Classes order; {multiclassPredicted ?? "PredictedLabel is the highest-scoring class"}";
            return Calibration is { IsMulticlass: true, Temperature: { } t }
                ? reads + Invariant($". To get the probabilities mloop predict reports, apply `calibration` (temperature {t:R}): raise each class's probability (at least 1e-12) to the power 1/{t:R} and divide by their sum — the highest-scoring class does not change")
                : reads;
        }
        return "Score is the predicted value";
    }

    private string DescribeBinary()
    {
        const string graph = "The graph's Probability is the trained model's own probability of the second class and its PredictedLabel is true (the second class) when Score > 0";
        var calibration = Calibration is { IsMulticlass: false } c ? c : null;
        if (calibration is null && Threshold is null)
            return graph;

        var steps = new List<string>();
        if (calibration is not null)
        {
            steps.Add(calibration.Method == "platt"
                ? Invariant($"map Probability through `calibration` (platt: with p held to [1e-7, 1 - 1e-7], p' = 1 / (1 + exp(-({calibration.A:R} * ln(p / (1 - p)) + {calibration.B:R}))))")
                : "map Probability through `calibration` (isotonic: linear between `knots` and their `values`, held flat outside them)");
        }
        var probability = calibration is null ? "Probability" : "the calibrated probability";
        steps.Add(Threshold is { } t
            ? Invariant($"answer the second class when {probability} >= {t.Threshold:R} (the decision threshold, `decisionThreshold`)")
            : $"answer the second class when {probability} >= 0.5");
        return $"{graph}. To answer as mloop predict does: {string.Join(", then ", steps)}";
    }

    private static string Invariant(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);
}
