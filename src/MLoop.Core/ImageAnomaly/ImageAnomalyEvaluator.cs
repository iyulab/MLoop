namespace MLoop.Core.ImageAnomaly;

/// <summary>What a set of scored normal and anomalous images says about how well the scores tell them apart.</summary>
/// <param name="Normals">How many normal images.</param>
/// <param name="Anomalies">How many anomalous images.</param>
/// <param name="Auroc">The chance that an anomalous image scores above a normal one (ties count half).</param>
/// <param name="MissedAt5">Anomalies at or below the score that lets 5 % of the normal images through as false alarms.</param>
/// <param name="MissedAt10">The same at 10 %.</param>
public sealed record AnomalyQuality(int Normals, int Anomalies, double Auroc, int MissedAt5, int MissedAt10)
{
    /// <summary>The metrics as a run records them.</summary>
    public Dictionary<string, double> AsMetrics() => new()
    {
        ["auroc"] = Auroc,
        ["anomalies_missed_at_5pct_false_alarms"] = MissedAt5,
        ["anomalies_missed_at_10pct_false_alarms"] = MissedAt10,
        ["normal_images"] = Normals,
        ["anomalous_images"] = Anomalies,
    };
}

/// <summary>
/// How well an image-anomaly model's scores separate normal images from anomalous ones: the AUROC, and — against a
/// threshold — how many anomalies a given false-alarm rate on the normal images lets through, which is how the cost of
/// missing a defect is usually counted (a rate of false alarms a line can bear, then the defects left).
/// </summary>
public static class ImageAnomalyEvaluator
{
    /// <summary>The names of a subfolder of normal images (any case).</summary>
    public static readonly IReadOnlyList<string> NormalFolderNames = ["normal", "good", "ok"];

    /// <summary>The quality of <paramref name="normalScores"/> against <paramref name="anomalyScores"/>.</summary>
    /// <exception cref="ArgumentException">Either set is empty: nothing to tell apart.</exception>
    public static AnomalyQuality Evaluate(IReadOnlyList<double> normalScores, IReadOnlyList<double> anomalyScores)
    {
        if (normalScores.Count == 0 || anomalyScores.Count == 0)
            throw new ArgumentException("Telling normal images from anomalous ones needs both: images of each.");
        return new AnomalyQuality(normalScores.Count, anomalyScores.Count, Auroc(normalScores, anomalyScores),
            Missed(normalScores, anomalyScores, 0.05), Missed(normalScores, anomalyScores, 0.10));
    }

    /// <summary>The share of (anomalous, normal) pairs the anomalous image scores higher in, ties half.</summary>
    public static double Auroc(IReadOnlyList<double> normalScores, IReadOnlyList<double> anomalyScores)
    {
        var sortedNormals = normalScores.Order().ToArray();
        double wins = 0;
        foreach (var anomaly in anomalyScores)
        {
            var below = LowerBound(sortedNormals, anomaly);               // normals strictly below
            var notAbove = UpperBound(sortedNormals, anomaly);            // normals at or below
            wins += below + (notAbove - below) / 2.0;
        }
        return wins / ((double)normalScores.Count * anomalyScores.Count);
    }

    /// <summary>
    /// How many anomalies score at or below the threshold that lets at most <paramref name="falseAlarmRate"/> of the normal
    /// images score above it — the rate is a count, rounded down: of 20 normal images 5 % is one false alarm.
    /// </summary>
    public static int Missed(IReadOnlyList<double> normalScores, IReadOnlyList<double> anomalyScores, double falseAlarmRate)
    {
        var allowed = (int)Math.Floor(falseAlarmRate * normalScores.Count + 1e-9);
        var descending = normalScores.OrderByDescending(s => s).ToArray();
        var threshold = descending[Math.Min(allowed, descending.Length - 1)];
        // Normal images strictly above the threshold are the false alarms: at most `allowed` of them.
        return anomalyScores.Count(a => a <= threshold);
    }

    private static int LowerBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) / 2;
            if (sorted[mid] < value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    private static int UpperBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) / 2;
            if (sorted[mid] <= value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
}
