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
/// <remarks>
/// The file is streamed, never held: memory follows the sample, not the file — the point of sampling a file too large
/// to profile whole. Random and stratified samples are drawn by reservoir sampling (one pass; stratified counts the
/// classes in a first pass), and every sample keeps the rows in their file order.
/// </remarks>
public sealed class CsvSampler
{
    private readonly CsvHelperImpl _csvHelper = new();
    private readonly Random _random;

    /// <summary>Creates a sampler; <paramref name="seed"/> makes random and stratified samples repeatable.</summary>
    public CsvSampler(int? seed = null)
    {
        _random = seed is { } s ? new Random(s) : Random.Shared;
    }

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

        var (headers, sampled, totalRows) = strategy switch
        {
            CsvSamplingStrategy.Head => await HeadAsync(inputPath, rows, cancellationToken).ConfigureAwait(false),
            CsvSamplingStrategy.Stratified => await StratifiedAsync(inputPath, rows, labelColumn!, cancellationToken).ConfigureAwait(false),
            _ => await RandomAsync(inputPath, rows, cancellationToken).ConfigureAwait(false)
        };

        if (totalRows == 0)
            throw new InvalidOperationException("Input CSV file is empty (no data rows).");

        await _csvHelper.WriteRowsAsync(outputPath, headers, sampled, cancellationToken).ConfigureAwait(false);

        return new CsvSamplingResult(
            OutputPath: Path.GetFullPath(outputPath),
            SampledCount: sampled.Count,
            TotalRows: totalRows,
            StrategyUsed: strategy);
    }

    // Head reads on past its rows only to count the file's — TotalRows is reported — and keeps nothing more.
    private async Task<(IReadOnlyList<string>, List<string[]>, int)> HeadAsync(string path, int size, CancellationToken ct)
    {
        var taken = new List<string[]>();
        var seen = 0;
        var headers = await _csvHelper.ReadRowsAsync(path, row =>
        {
            if (seen++ < size)
                taken.Add(row);
            return true;
        }, cancellationToken: ct).ConfigureAwait(false);
        return (headers, taken, seen);
    }

    private async Task<(IReadOnlyList<string>, List<string[]>, int)> RandomAsync(string path, int size, CancellationToken ct)
    {
        var reservoir = new Reservoir(size, _random);
        var headers = await _csvHelper.ReadRowsAsync(path, row => { reservoir.Offer(row); return true; },
            cancellationToken: ct).ConfigureAwait(false);
        return (headers, reservoir.InFileOrder(), reservoir.Seen);
    }

    private async Task<(IReadOnlyList<string>, List<string[]>, int)> StratifiedAsync(
        string path, int size, string labelColumn, CancellationToken ct)
    {
        // First pass: how many rows each class has.
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var labelIndex = -1;
        var headers = await _csvHelper.ReadRowsAsync(path, row =>
        {
            var key = Key(row[labelIndex]);
            counts[key] = counts.GetValueOrDefault(key) + 1;
            return true;
        }, onHeaders: h =>
        {
            labelIndex = IndexOf(h, labelColumn);
            if (labelIndex < 0)
                throw new ArgumentException(
                    $"Label column '{labelColumn}' not found in CSV. Available columns: {string.Join(", ", h)}",
                    nameof(labelColumn));
        }, cancellationToken: ct).ConfigureAwait(false);

        var total = counts.Values.Sum();
        if (total == 0)
            return (headers, [], 0);

        // Second pass: a reservoir per class, sized in proportion to the class.
        var reservoirs = Quotas(counts, Math.Min(size, total))
            .ToDictionary(q => q.Key, q => new Reservoir(q.Value, _random), StringComparer.Ordinal);
        var index = 0;
        await _csvHelper.ReadRowsAsync(path, row =>
        {
            if (reservoirs.TryGetValue(Key(row[labelIndex]), out var reservoir))
                reservoir.Offer(row, index);
            index++;
            return true;
        }, cancellationToken: ct).ConfigureAwait(false);

        var sampled = reservoirs.Values.SelectMany(r => r.Kept).OrderBy(k => k.Index).Select(k => k.Row).ToList();
        return (headers, sampled, total);
    }

    private static string Key(string label) => string.IsNullOrEmpty(label) ? "_missing_" : label;

    private static int IndexOf(IReadOnlyList<string> headers, string name)
    {
        for (var i = 0; i < headers.Count; i++)
            if (headers[i] == name)
                return i;
        return -1;
    }

    /// <summary>
    /// Each class's share of <paramref name="size"/> rows — the class's proportion, by largest remainder, so the shares
    /// add up to the sample — with at least one row for every class when the sample has a row for each; a share never
    /// exceeds the class's rows.
    /// </summary>
    internal static Dictionary<string, int> Quotas(IReadOnlyDictionary<string, int> counts, int size)
    {
        double total = counts.Values.Sum();
        var order = counts.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
        var exact = order.ToDictionary(k => k, k => size * (counts[k] / total), StringComparer.Ordinal);
        var quotas = order.ToDictionary(k => k, k => Math.Min(counts[k], (int)Math.Floor(exact[k])), StringComparer.Ordinal);

        if (size >= order.Count)
            foreach (var k in order.Where(k => quotas[k] == 0))
                quotas[k] = 1;

        // The floors of one row can overshoot: take back from the shares furthest above their exact value.
        while (quotas.Values.Sum() > size)
        {
            var largest = order.Where(k => quotas[k] > 1).MaxBy(k => quotas[k] - exact[k])!;
            quotas[largest]--;
        }
        // Rows rounding left over go to the largest remainders among classes with rows to give.
        while (quotas.Values.Sum() < size)
        {
            var next = order.Where(k => quotas[k] < counts[k]).MaxBy(k => exact[k] - quotas[k]);
            if (next is null)
                break;
            quotas[next]++;
        }
        return quotas.Where(q => q.Value > 0).ToDictionary(q => q.Key, q => q.Value, StringComparer.Ordinal);
    }

    /// <summary>Algorithm R: a uniform sample of fixed size from a stream of unknown length.</summary>
    private sealed class Reservoir(int size, Random random)
    {
        private readonly List<(int Index, string[] Row)> _kept = new(Math.Min(size, 1 << 16));

        public int Seen { get; private set; }

        public IEnumerable<(int Index, string[] Row)> Kept => _kept;

        public void Offer(string[] row) => Offer(row, Seen);

        public void Offer(string[] row, int index)
        {
            Seen++;
            if (_kept.Count < size)
            {
                _kept.Add((index, row));
                return;
            }
            var slot = random.NextInt64(Seen);
            if (slot < size)
                _kept[(int)slot] = (index, row);
        }

        public List<string[]> InFileOrder() => [.. _kept.OrderBy(k => k.Index).Select(k => k.Row)];
    }
}
