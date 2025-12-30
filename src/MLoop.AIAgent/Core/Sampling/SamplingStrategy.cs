namespace MLoop.AIAgent.Core.Sampling;

/// <summary>
/// Sampling method for incremental preprocessing.
/// </summary>
public enum SamplingMethod
{
    /// <summary>Pure random sampling.</summary>
    Random,

    /// <summary>Proportional representation across categories.</summary>
    Stratified,

    /// <summary>Every Nth record (systematic sampling).</summary>
    Systematic,

    /// <summary>Group-based sampling for clustered data.</summary>
    ClusterBased
}

/// <summary>
/// Configuration for sampling at each stage.
/// </summary>
public record StageSamplingConfig(
    int Stage,
    double SampleRate,
    int MinSampleSize,
    string Purpose);

/// <summary>
/// Manages progressive sampling in incremental preprocessing.
/// Controls sample sizes and selection methods across the 5-stage workflow.
/// </summary>
/// <remarks>
/// Named SamplingManager to avoid conflict with ironbees SamplingStrategy enum.
/// Ironbees defines the schema (SamplingStrategy enum), this class provides execution.
/// </remarks>
public class SamplingManager
{
    /// <summary>
    /// Stage-based sample configurations.
    /// </summary>
    public static readonly StageSamplingConfig[] StageConfigs =
    [
        new(1, 0.001, 100, "Initial Exploration"),
        new(2, 0.005, 500, "Pattern Expansion"),
        new(3, 0.015, 1500, "HITL Decision"),
        new(4, 0.025, 2500, "Confidence Checkpoint"),
        new(5, 1.0, int.MaxValue, "Bulk Processing")
    ];

    /// <summary>
    /// Gets or sets the sampling method.
    /// </summary>
    public SamplingMethod Method { get; set; } = SamplingMethod.Stratified;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the stratification column for stratified sampling.
    /// </summary>
    public string? StratificationColumn { get; set; }

    private readonly Random _random;

    public SamplingManager(int? seed = null)
    {
        RandomSeed = seed;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Gets the sample size for a given stage and total record count.
    /// </summary>
    /// <param name="totalRecords">Total number of records in the dataset.</param>
    /// <param name="stage">Stage number (1-5).</param>
    /// <returns>Number of records to sample.</returns>
    public int GetSampleSize(int totalRecords, int stage)
    {
        if (stage < 1 || stage > StageConfigs.Length)
            throw new ArgumentOutOfRangeException(nameof(stage), "Stage must be between 1 and 5");

        var config = StageConfigs[stage - 1];

        // Stage 5 is always full dataset
        if (stage == 5)
            return totalRecords;

        // Calculate sample size based on rate
        var calculatedSize = (int)(totalRecords * config.SampleRate);

        // Ensure minimum sample size (but not more than total)
        return Math.Min(totalRecords, Math.Max(calculatedSize, config.MinSampleSize));
    }

    /// <summary>
    /// Gets a sample of records from the data.
    /// </summary>
    /// <typeparam name="T">Record type.</typeparam>
    /// <param name="data">Source data enumerable.</param>
    /// <param name="stage">Current stage (1-5).</param>
    /// <param name="totalRecords">Total number of records (for size calculation).</param>
    /// <returns>Sampled records.</returns>
    public IEnumerable<T> GetSample<T>(IEnumerable<T> data, int stage, int totalRecords)
    {
        var sampleSize = GetSampleSize(totalRecords, stage);

        return Method switch
        {
            SamplingMethod.Random => GetRandomSample(data, sampleSize),
            SamplingMethod.Systematic => GetSystematicSample(data, sampleSize, totalRecords),
            SamplingMethod.Stratified => GetRandomSample(data, sampleSize), // TODO: Implement stratified
            SamplingMethod.ClusterBased => GetRandomSample(data, sampleSize), // TODO: Implement cluster
            _ => GetRandomSample(data, sampleSize)
        };
    }

    /// <summary>
    /// Gets indices for sampling (useful for row-based access).
    /// </summary>
    /// <param name="totalRecords">Total number of records.</param>
    /// <param name="stage">Current stage (1-5).</param>
    /// <returns>Set of indices to sample.</returns>
    public HashSet<int> GetSampleIndices(int totalRecords, int stage)
    {
        var sampleSize = GetSampleSize(totalRecords, stage);
        var indices = new HashSet<int>();

        if (sampleSize >= totalRecords)
        {
            // Return all indices
            for (int i = 0; i < totalRecords; i++)
                indices.Add(i);
            return indices;
        }

        return Method switch
        {
            SamplingMethod.Systematic => GetSystematicIndices(totalRecords, sampleSize),
            _ => GetRandomIndices(totalRecords, sampleSize)
        };
    }

    /// <summary>
    /// Gets the cumulative sample size up to and including the specified stage.
    /// </summary>
    public int GetCumulativeSampleSize(int totalRecords, int stage)
    {
        int cumulative = 0;
        for (int s = 1; s <= stage; s++)
        {
            cumulative = GetSampleSize(totalRecords, s);
        }
        return cumulative;
    }

    /// <summary>
    /// Gets the stage configuration.
    /// </summary>
    public static StageSamplingConfig GetStageConfig(int stage)
    {
        if (stage < 1 || stage > StageConfigs.Length)
            throw new ArgumentOutOfRangeException(nameof(stage));
        return StageConfigs[stage - 1];
    }

    private IEnumerable<T> GetRandomSample<T>(IEnumerable<T> data, int sampleSize)
    {
        var list = data.ToList();
        if (sampleSize >= list.Count)
            return list;

        // Fisher-Yates shuffle for first sampleSize elements
        for (int i = 0; i < sampleSize; i++)
        {
            int j = _random.Next(i, list.Count);
            (list[i], list[j]) = (list[j], list[i]);
        }

        return list.Take(sampleSize);
    }

    private static IEnumerable<T> GetSystematicSample<T>(IEnumerable<T> data, int sampleSize, int totalRecords)
    {
        if (sampleSize >= totalRecords)
            return data;

        var interval = (double)totalRecords / sampleSize;
        var list = data.ToList();
        var result = new List<T>(sampleSize);

        for (int i = 0; i < sampleSize; i++)
        {
            var index = (int)(i * interval);
            if (index < list.Count)
                result.Add(list[index]);
        }

        return result;
    }

    private HashSet<int> GetRandomIndices(int totalRecords, int sampleSize)
    {
        var indices = new HashSet<int>();
        while (indices.Count < sampleSize)
        {
            indices.Add(_random.Next(totalRecords));
        }
        return indices;
    }

    private static HashSet<int> GetSystematicIndices(int totalRecords, int sampleSize)
    {
        var indices = new HashSet<int>();
        var interval = (double)totalRecords / sampleSize;

        for (int i = 0; i < sampleSize; i++)
        {
            var index = (int)(i * interval);
            if (index < totalRecords)
                indices.Add(index);
        }

        return indices;
    }
}
