using Microsoft.ML;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// A classifier's <c>Score</c> is read as class probabilities by every path — the response's probabilities and
/// confidence, the evaluator's log loss, a calibration. The NAS-BERT text classifier gives logits there; they are
/// turned into probabilities when the model is loaded, and ML.NET's own classifiers are left as they are.
/// </summary>
public class ClassScoresTests
{
    [Fact]
    public void Softmax_GivesProbabilities_InTheLogitsOrder()
    {
        var p = ClassScores.Softmax([-0.0278f, 0.0321f, 2f]);

        Assert.Equal(1.0, p.Sum(), 5);
        Assert.All(p, v => Assert.InRange(v, 0f, 1f));
        Assert.True(p[2] > p[1] && p[1] > p[0]);
    }

    [Fact]
    public void Softmax_OfLargeLogits_DoesNotOverflow()
    {
        var p = ClassScores.Softmax([1000f, 999f]);

        Assert.Equal(1.0, p.Sum(), 5);
        Assert.Equal(1 / (1 + Math.Exp(-1)), p[0], 5);
    }

    [Fact]
    public void AnMlNetClassifier_ScoresWithProbabilities_AndIsLeftAsItIs()
    {
        var ml = new MLContext(1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Row { X = 1, Label = "a" }, new Row { X = 2, Label = "a" }, new Row { X = 8, Label = "b" }, new Row { X = 9, Label = "b" },
        });
        var model = ml.Transforms.Conversion.MapValueToKey("Label")
            .Append(ml.Transforms.Concatenate("Features", nameof(Row.X)))
            .Append(ml.MulticlassClassification.Trainers.SingleThreadSdcaMaximumEntropy())
            .Fit(data);

        Assert.False(ClassScores.AreLogits(model));
        Assert.Same(model, ClassScores.AsProbabilities(ml, model));
    }

    private sealed class Row
    {
        public float X { get; set; }
        public string Label { get; set; } = "";
    }
}
