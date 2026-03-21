namespace MLoop.Core.Data;

/// <summary>
/// Sampling strategy for CSV files.
/// </summary>
public enum CsvSamplingStrategy
{
    /// <summary>Take first N rows (head).</summary>
    Head,

    /// <summary>Random sampling without replacement.</summary>
    Random,

    /// <summary>Stratified sampling preserving label column distribution.</summary>
    Stratified
}

/// <summary>
/// Result of a CSV sampling operation.
/// </summary>
public record CsvSamplingResult(
    string OutputPath,
    int SampledCount,
    int TotalRows,
    CsvSamplingStrategy StrategyUsed);

/// <summary>
/// General-purpose CSV file sampler with encoding detection.
/// Supports head, random, and stratified sampling strategies.
/// </summary>
public sealed class CsvSampler
{
    private readonly CsvHelperImpl _csvHelper = new();

    /// <summary>
    /// Samples rows from a CSV file and writes the result to an output file.
    /// </summary>
    /// <param name="inputPath">Source CSV file path</param>
    /// <param name="outputPath">Destination CSV file path</param>
    /// <param name="rows">Number of rows to sample</param>
    /// <param name="strategy">Sampling strategy</param>
    /// <param name="labelColumn">Label column name (required for stratified sampling)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task<CsvSamplingResult> SampleAsync(
        string inputPath,
        string outputPath,
        int rows,
        CsvSamplingStrategy strategy = CsvSamplingStrategy.Random,
        string? labelColumn = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(inputPath))
            throw new FileNotFoundException($"Input CSV file not found: {inputPath}");

        if (rows <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Row count must be positive.");

        if (strategy == CsvSamplingStrategy.Stratified && string.IsNullOrWhiteSpace(labelColumn))
            throw new ArgumentException("Label column is required for stratified sampling.", nameof(labelColumn));

        var data = await _csvHelper.ReadAsync(inputPath, cancellationToken: cancellationToken).ConfigureAwait(false);

        if (data.Count == 0)
            throw new InvalidOperationException("Input CSV file is empty (no data rows).");

        // Validate label column exists for stratified sampling
        if (strategy == CsvSamplingStrategy.Stratified && !data[0].ContainsKey(labelColumn!))
            throw new ArgumentException(
                $"Label column '{labelColumn}' not found in CSV. Available columns: {string.Join(", ", data[0].Keys)}",
                nameof(labelColumn));

        var actualSize = Math.Min(rows, data.Count);
        var sampled = strategy switch
        {
            CsvSamplingStrategy.Head => HeadSample(data, actualSize),
            CsvSamplingStrategy.Random => RandomSample(data, actualSize),
            CsvSamplingStrategy.Stratified => StratifiedSample(data, actualSize, labelColumn!),
            _ => RandomSample(data, actualSize)
        };

        var columnOrder = data[0].Keys.ToList();
        await _csvHelper.WriteAsync(outputPath, sampled, columnOrder, cancellationToken).ConfigureAwait(false);

        return new CsvSamplingResult(
            OutputPath: Path.GetFullPath(outputPath),
            SampledCount: sampled.Count,
            TotalRows: data.Count,
            StrategyUsed: strategy);
    }

    private static List<Dictionary<string, string>> HeadSample(
        List<Dictionary<string, string>> data, int size)
    {
        return data.Take(size).ToList();
    }

    private static List<Dictionary<string, string>> RandomSample(
        List<Dictionary<string, string>> data, int size)
    {
        return data.OrderBy(_ => Random.Shared.Next()).Take(size).ToList();
    }

    private static List<Dictionary<string, string>> StratifiedSample(
        List<Dictionary<string, string>> data, int size, string labelColumn)
    {
        var groups = data.GroupBy(row =>
            row.TryGetValue(labelColumn, out var v) ? v : "_missing_")
            .ToDictionary(g => g.Key, g => g.ToList());

        if (groups.Count == 0)
            return new List<Dictionary<string, string>>();

        var result = new List<Dictionary<string, string>>();
        var totalCount = data.Count;
        var remaining = size;

        foreach (var (_, group) in groups.OrderByDescending(g => g.Value.Count))
        {
            var proportion = (double)group.Count / totalCount;
            var groupSize = Math.Max(1, (int)Math.Round(size * proportion));
            groupSize = Math.Min(groupSize, remaining);
            groupSize = Math.Min(groupSize, group.Count);

            if (groupSize > 0)
            {
                result.AddRange(group.OrderBy(_ => Random.Shared.Next()).Take(groupSize));
                remaining -= groupSize;
            }

            if (remaining <= 0)
                break;
        }

        // Fill remaining if rounding left gaps
        if (remaining > 0)
        {
            var usedSet = new HashSet<Dictionary<string, string>>(result, ReferenceEqualityComparer.Instance);
            var unused = data.Where(r => !usedSet.Contains(r)).ToList();
            result.AddRange(unused.OrderBy(_ => Random.Shared.Next()).Take(remaining));
        }

        return result;
    }
}
