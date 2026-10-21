using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Models;

public class DecisionThresholdTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-threshold-" + Guid.NewGuid().ToString("N"));

    public DecisionThresholdTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private static (double, bool, double) R(double p, bool positive, double w = 1) => (p, positive, w);

    [Fact]
    public void TheLowestThresholdThatReachesThePrecision_IsChosen_KeepingTheMostRecall()
    {
        // Descending: 0.9+ 0.8+ 0.7- 0.6+ 0.5- 0.4+ — precision 1, 1, .67, .75, .6, .67
        var rows = new[] { R(0.9, true), R(0.8, true), R(0.7, false), R(0.6, true), R(0.5, false), R(0.4, true) };

        Assert.Equal((0.8, 1.0, 0.5), DecisionThreshold.Choose(rows, 1.0));
        var chosen = DecisionThreshold.Choose(rows, 0.75)!.Value;
        Assert.Equal(0.6, chosen.Threshold);       // .75 at 0.6 keeps 3 of 4 positives
        Assert.Equal(0.75, chosen.Recall);
    }

    [Fact]
    public void RowsOfOneProbability_AreAnsweredAlike_SoNoThresholdSplitsThem()
    {
        // At 0.8 one positive and one negative tie: a threshold there takes both (precision .67), never just the positive.
        var rows = new[] { R(0.9, true), R(0.8, true), R(0.8, false) };

        Assert.Equal(0.9, DecisionThreshold.Choose(rows, 0.9)!.Value.Threshold);
    }

    [Fact]
    public void Weights_CountInThePrecision()
    {
        var rows = new[] { R(0.9, true), R(0.8, false, w: 9), R(0.7, true) };

        Assert.Equal(0.9, DecisionThreshold.Choose(rows, 0.5)!.Value.Threshold);   // the heavy negative sinks 0.8 and 0.7
    }

    [Fact]
    public void WhenNoThresholdReachesThePrecision_ThereIsNone()
    {
        Assert.Null(DecisionThreshold.Choose([R(0.9, false), R(0.8, true)], 0.9));
        Assert.Null(DecisionThreshold.Choose([R(0.9, false)], 0.5));   // no positive at all
    }

    private sealed class Scored
    {
        public float Probability { get; set; }
        public bool PredictedLabel { get; set; }
    }

    [Fact]
    public void Applied_ItAnswersPositiveFromTheThresholdOn()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Scored { Probability = 0.9f, PredictedLabel = true },
            new Scored { Probability = 0.6f, PredictedLabel = true },
            new Scored { Probability = 0.3f, PredictedLabel = false },
        });
        var identity = ml.Transforms.CopyColumns("Probability2", "Probability").Fit(data);
        var threshold = new DecisionThreshold(0.7, 0.9, 1, 0.5, 3, "f.csv", null, DateTimeOffset.UtcNow);

        var answered = threshold.Apply(ml, identity).Transform(data);

        Assert.Equal([true, false, false], answered.GetColumn<bool>("PredictedLabel").ToArray());
    }

    [Fact]
    public void ItIsReadFromBesideTheModel_AndAbsentMeansNone()
    {
        var model = Path.Combine(_dir, ExperimentLayout.ModelFileName);
        Assert.Null(DecisionThreshold.ReadBeside(model));
        var threshold = new DecisionThreshold(0.42, 0.9, 0.91, 0.6, 120, "recheck.csv", "w", DateTimeOffset.UtcNow);

        threshold.WriteBeside(model);

        Assert.Equal(threshold, DecisionThreshold.ReadBeside(model));
        Assert.Equal(Path.Combine(_dir, ExperimentLayout.DecisionThresholdFileName), DecisionThreshold.PathBeside(model));
    }
}
