using MLoop.Core.Export;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Export;

/// <summary>
/// <see cref="OnnxDecisionRecipe"/> — the manifest's decision sentence and the arithmetic
/// <c>--verify</c> replays come from one place, in <c>mloop predict</c>'s order: calibration, then threshold.
/// </summary>
public sealed class OnnxDecisionRecipeTests
{
    private static ProbabilityCalibration Platt(double a, double b) =>
        new("platt", a, b, null, null, 100, "f.csv", null, null, null, DateTimeOffset.UnixEpoch);

    private static DecisionThreshold Threshold(double t) =>
        new(t, 0.9, 0.9, 0.5, 100, "f.csv", null, DateTimeOffset.UnixEpoch);

    [Fact]
    public void Binary_WithoutLayers_IsTheGraphsOwnAnswer()
    {
        var recipe = new OnnxDecisionRecipe("binary-classification", null, null);

        Assert.False(recipe.HasLayers);
        Assert.Equal((0.3f, true), recipe.AnswerBinary(0.3f, true));
        Assert.Contains("Score > 0", recipe.Describe());
        Assert.DoesNotContain("mloop predict", recipe.Describe());
    }

    [Fact]
    public void Binary_Calibrated_AnswersAtHalfOfTheCalibratedProbability_NotAtTheGraphsLabel()
    {
        // a = 1, b = 2 lifts every probability: 0.3 → σ(logit(0.3) + 2) ≈ 0.76.
        var recipe = new OnnxDecisionRecipe("binary-classification", Platt(1, 2), null);

        var (probability, label) = recipe.AnswerBinary(0.3f, false);

        Assert.True(recipe.HasLayers);
        Assert.Equal(0.76, probability, 2);
        Assert.True(label);
        var sentence = recipe.Describe();
        Assert.Contains("`calibration`", sentence);
        Assert.Contains("the calibrated probability >= 0.5", sentence);
    }

    [Fact]
    public void Binary_CalibratedWithAThreshold_ComparesTheCalibratedProbability()
    {
        var recipe = new OnnxDecisionRecipe("binary-classification", Platt(1, 2), Threshold(0.8));

        Assert.False(recipe.AnswerBinary(0.3f, true).Label);   // 0.76 < 0.8
        Assert.True(recipe.AnswerBinary(0.4f, false).Label);   // ≈ 0.83
        var sentence = recipe.Describe();
        Assert.True(sentence.IndexOf("`calibration`", StringComparison.Ordinal) < sentence.IndexOf("0.8", StringComparison.Ordinal));
        Assert.Contains("the calibrated probability >= 0.8", sentence);
    }

    [Fact]
    public void Binary_ThresholdAlone_AppliesToTheGraphsProbability()
    {
        var recipe = new OnnxDecisionRecipe("binary-classification", null, Threshold(0.25));

        Assert.Equal((0.3f, true), recipe.AnswerBinary(0.3f, false));
        Assert.Contains("Probability >= 0.25", recipe.Describe());
    }

    [Fact]
    public void Multiclass_Temperature_TempersTheScoresAndKeepsTheAnswer()
    {
        var temperature = new ProbabilityCalibration("temperature", null, null, null, null, 100, "f.csv", null, null, null,
            DateTimeOffset.UnixEpoch, Temperature: 2);
        var recipe = new OnnxDecisionRecipe("multiclass-classification", temperature, null);

        var tempered = recipe.AnswerMulticlass([0.7f, 0.2f, 0.1f]);

        Assert.True(recipe.HasLayers);
        Assert.Equal(1.0, tempered.Sum(), 5);
        Assert.True(tempered[0] < 0.7f && tempered[0] == tempered.Max());
        Assert.Contains("temperature 2", recipe.Describe("PredictedLabel is the name of the highest-scoring class"));
    }

    [Fact]
    public void ABinaryLayerOnAMulticlassModel_IsNotARule()
    {
        Assert.False(new OnnxDecisionRecipe("multiclass-classification", Platt(1, 0), null).HasLayers);
        Assert.False(new OnnxDecisionRecipe("regression", null, Threshold(0.5)).HasLayers);
    }
}
