using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using MLoop.Core.Prediction;
using MLoop.DataStore.Interfaces;

namespace MLoop.DataStore.Services;

/// <summary>
/// Filesystem-based data sampler that creates retraining datasets from predictions and feedback.
/// Supports Random, Recent, FeedbackPriority, Stratified and LowConfidence strategies.
/// </summary>
public sealed class FileDataSampler : IDataSampler
{
    private readonly string _logsDirectory;
    private readonly FileFeedbackCollector _feedback;

    public FileDataSampler(string baseDirectory)
    {
        _logsDirectory = Path.Combine(baseDirectory, ".mloop", "logs");
        // Feedback is read through its own store — one reader of the format, so a field it gains
        // (a weight) reaches sampling without a second parser to update.
        _feedback = new FileFeedbackCollector(baseDirectory);
    }

    /// <inheritdoc />
    public async Task<SamplingResult> SampleAsync(
        string modelName,
        int sampleSize,
        SamplingStrategy strategy,
        string outputPath,
        RetrainingColumns columns,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(columns);
        ArgumentException.ThrowIfNullOrWhiteSpace(columns.Label);

        var feedback = await LoadFeedbackAsync(modelName, cancellationToken).ConfigureAwait(false);

        // Feedback can be recorded more than once for the same prediction (RecordFeedbackAsync appends — e.g. a
        // user correcting a label): the latest stands.
        var feedbackDict = FeedbackEntry.LatestPerPrediction(feedback)
            .ToDictionary(f => f.PredictionId);

        // The logged predictions are read as a stream and only the sample is held: a model logging 80,000
        // predictions a day keeps millions over its retention period, and every one used to be held with its
        // inputs parsed before the strategy chose a few hundred.
        var (picked, total) = await ApplyStrategyAsync(modelName, Math.Max(sampleSize, 0), strategy, feedbackDict, cancellationToken)
            .ConfigureAwait(false);

        if (total == 0)
        {
            throw new InvalidOperationException(
                $"No predictions found for model '{modelName}'. " +
                "Log predictions using 'mloop predict --log' first.");
        }

        var sampled = picked.Select(p => p.Join(feedbackDict)).ToList();

        // A weight column is written when the model trains with one (its rows must carry it), or when
        // feedback carries weights the model has not been told about yet.
        var weightColumn = columns.Weight
            ?? (sampled.Any(s => s.Feedback?.Weight is not null) ? DefaultWeightColumn(sampled, columns.Label) : null);

        // Logged inputs carry the client's spelling; the file is written in the model's.
        var spelling = columns.Inputs.Distinct(StringComparer.OrdinalIgnoreCase)
            .ToDictionary(c => c, c => c, StringComparer.OrdinalIgnoreCase);
        sampled = sampled
            .Select(s => s with
            {
                Input = s.Input
                    .GroupBy(kv => spelling.GetValueOrDefault(kv.Key, kv.Key))
                    .ToDictionary(g => g.Key, g => g.First().Value)
            })
            .ToList();

        await WriteCsvAsync(sampled, outputPath, columns.Label, weightColumn, cancellationToken).ConfigureAwait(false);

        return new SamplingResult(
            OutputPath: outputPath,
            SampledCount: sampled.Count,
            TotalAvailable: (int)Math.Min(total, int.MaxValue),
            StrategyUsed: strategy,
            CreatedAt: DateTimeOffset.UtcNow,
            LabeledCount: sampled.Count(s => s.Feedback is not null),
            LabelColumn: columns.Label,
            WeightColumn: weightColumn);
    }

    /// <inheritdoc />
    public async Task<SamplingStatistics> GetStatisticsAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var feedback = await LoadFeedbackAsync(modelName, cancellationToken).ConfigureAwait(false);
        var feedbackIds = new HashSet<string>(feedback.Select(f => f.PredictionId));

        long total = 0, withFeedback = 0, lowConfidence = 0;
        var oldest = DateTimeOffset.MaxValue;
        var newest = DateTimeOffset.MinValue;
        await foreach (var p in ReadPredictionsAsync(modelName, cancellationToken).ConfigureAwait(false))
        {
            total++;
            if (feedbackIds.Contains(p.PredictionId))
                withFeedback++;
            if (p.Confidence is < 0.7)
                lowConfidence++;
            if (p.Timestamp < oldest)
                oldest = p.Timestamp;
            if (p.Timestamp > newest)
                newest = p.Timestamp;
        }

        static int Clamp(long n) => (int)Math.Min(n, int.MaxValue);
        return total == 0
            ? new SamplingStatistics(
                ModelName: modelName,
                TotalPredictions: 0,
                PredictionsWithFeedback: 0,
                LowConfidenceCount: 0,
                OldestEntry: DateTimeOffset.MinValue,
                NewestEntry: DateTimeOffset.MinValue)
            : new SamplingStatistics(
                ModelName: modelName,
                TotalPredictions: Clamp(total),
                PredictionsWithFeedback: Clamp(withFeedback),
                LowConfidenceCount: Clamp(lowConfidence),
                OldestEntry: oldest,
                NewestEntry: newest);
    }

    /// <summary>
    /// The strategy's pick from the logged predictions, holding only what it may keep, and how many were logged.
    /// Each strategy keeps the rows a sort of the whole would have given: random = a uniform sample without
    /// replacement, recent = the newest, low confidence = the least confident, feedback priority = rows with
    /// feedback first (a uniform sample of them when they are more than the sample) then a uniform sample of the
    /// rest, stratified = per output value in proportion (a counting pass first).
    /// </summary>
    private async Task<(List<LoggedPrediction> Picked, long Total)> ApplyStrategyAsync(
        string modelName,
        int size,
        SamplingStrategy strategy,
        Dictionary<string, FeedbackEntry> feedback,
        CancellationToken cancellationToken)
    {
        var random = new Random();
        var predictions = ReadPredictionsAsync(modelName, cancellationToken);

        switch (strategy)
        {
            case SamplingStrategy.Recent:
                {
                    var newest = new TopEntries<LoggedPrediction, DateTimeOffset>(size, p => p.Timestamp);
                    var total = 0L;
                    await foreach (var p in predictions.ConfigureAwait(false))
                    {
                        total++;
                        newest.Add(p);
                    }
                    return (newest.LargestFirst(), total);
                }

            case SamplingStrategy.LowConfidence:
                {
                    // Rows without a confidence are not candidates; the least confident first.
                    var least = new TopEntries<LoggedPrediction, double>(size, p => -p.Confidence!.Value);
                    var total = 0L;
                    await foreach (var p in predictions.ConfigureAwait(false))
                    {
                        total++;
                        if (p.Confidence.HasValue)
                            least.Add(p);
                    }
                    return (least.LargestFirst(), total);
                }

            case SamplingStrategy.FeedbackPriority:
                {
                    var answered = new Reservoir<LoggedPrediction>(size, random);
                    var rest = new Reservoir<LoggedPrediction>(size, random);
                    await foreach (var p in predictions.ConfigureAwait(false))
                        (feedback.ContainsKey(p.PredictionId) ? answered : rest).Add(p);

                    var picked = answered.Kept.ToList();
                    picked.AddRange(rest.Kept.Take(size - picked.Count));
                    return (picked, answered.Seen + rest.Seen);
                }

            case SamplingStrategy.Stratified:
                return await StratifiedAsync(modelName, size, random, cancellationToken).ConfigureAwait(false);

            default:
                {
                    var reservoir = new Reservoir<LoggedPrediction>(size, random);
                    await foreach (var p in predictions.ConfigureAwait(false))
                        reservoir.Add(p);
                    return (reservoir.Kept, reservoir.Seen);
                }
        }
    }

    /// <summary>
    /// Stratified by output value: a first pass counts each value, the quotas follow the proportions — every value
    /// at least one row while the sample has room, largest values first — and a slot the rounding left over goes to
    /// a value with rows still unpicked, in proportion to how many; a second pass keeps a uniform sample of each
    /// value's quota.
    /// </summary>
    private async Task<(List<LoggedPrediction> Picked, long Total)> StratifiedAsync(
        string modelName, int size, Random random, CancellationToken cancellationToken)
    {
        var counts = new Dictionary<string, long>(StringComparer.Ordinal);
        var order = new List<string>();
        var total = 0L;
        await foreach (var p in ReadPredictionsAsync(modelName, cancellationToken).ConfigureAwait(false))
        {
            total++;
            var key = p.StratumKey;
            if (counts.TryGetValue(key, out var n))
                counts[key] = n + 1;
            else
            {
                counts[key] = 1;
                order.Add(key);
            }
        }

        if (total == 0)
            return ([], 0);

        var actualSize = (int)Math.Min(size, total);
        var quotas = new Dictionary<string, int>(StringComparer.Ordinal);
        var remaining = actualSize;
        foreach (var key in order.OrderByDescending(k => counts[k]))
        {
            if (remaining <= 0)
                break;
            var quota = (int)Math.Min(Math.Min(Math.Max(1, (long)Math.Round(actualSize * (double)counts[key] / total)), remaining), counts[key]);
            quotas[key] = quota;
            remaining -= quota;
        }

        // Left-over slots: each one to a value drawn in proportion to its unpicked rows — the distribution a
        // uniform draw from all unpicked rows gives.
        while (remaining > 0)
        {
            var spare = order.Select(k => (Key: k, Spare: counts[k] - quotas.GetValueOrDefault(k))).Where(s => s.Spare > 0).ToList();
            var totalSpare = spare.Sum(s => s.Spare);
            if (totalSpare == 0)
                break;
            var draw = random.NextInt64(totalSpare);
            foreach (var (key, free) in spare)
            {
                if (draw < free)
                {
                    quotas[key] = quotas.GetValueOrDefault(key) + 1;
                    break;
                }
                draw -= free;
            }
            remaining--;
        }

        var reservoirs = quotas.ToDictionary(q => q.Key, q => new Reservoir<LoggedPrediction>(q.Value, random), StringComparer.Ordinal);
        await foreach (var p in ReadPredictionsAsync(modelName, cancellationToken).ConfigureAwait(false))
            if (reservoirs.TryGetValue(p.StratumKey, out var reservoir))
                reservoir.Add(p);

        return ([.. order.Where(reservoirs.ContainsKey).SelectMany(k => reservoirs[k].Kept)], total);
    }

    /// <summary>
    /// The model's logged predictions, oldest day file first, one line at a time. The inputs are not parsed here —
    /// the line is kept, and parsed only for a row the strategy picks.
    /// </summary>
    private async IAsyncEnumerable<LoggedPrediction> ReadPredictionsAsync(
        string modelName,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var modelDir = Path.Combine(_logsDirectory, SanitizeFileName(modelName));
        if (!Directory.Exists(modelDir))
            yield break;

        foreach (var file in Directory.GetFiles(modelDir, "*.jsonl").Order(StringComparer.Ordinal))
        {
            if (cancellationToken.IsCancellationRequested)
                yield break;

            await foreach (var line in JsonLines.ReadAsync(file, cancellationToken).ConfigureAwait(false))
            {
                LoggedPrediction? prediction = null;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    if (root.TryGetProperty("id", out var idEl) && idEl.GetString() is { } id)
                    {
                        var output = root.TryGetProperty("output", out var outputEl)
                            ? GetObjectFromJson(outputEl)
                            : null;

                        var confidence = root.TryGetProperty("confidence", out var confEl) && confEl.ValueKind == JsonValueKind.Number
                            ? confEl.GetDouble()
                            : (double?)null;

                        var timestamp = root.TryGetProperty("timestamp", out var tsEl)
                            ? tsEl.GetDateTimeOffset()
                            : DateTimeOffset.MinValue;

                        prediction = new LoggedPrediction(id, output, confidence, timestamp, line);
                    }
                }
                catch (JsonException)
                {
                    // Skip malformed entries
                }

                if (prediction is not null)
                    yield return prediction;
            }
        }
    }

    private Task<IReadOnlyList<FeedbackEntry>> LoadFeedbackAsync(
        string modelName,
        CancellationToken cancellationToken) =>
        _feedback.GetFeedbackAsync(modelName, limit: int.MaxValue, cancellationToken: cancellationToken);

    /// <summary>
    /// <see cref="RetrainingColumns.DefaultWeight"/>, unless a logged input or the label already holds
    /// that name — then <c>feedback_weight</c>, so the weight never overwrites a feature.
    /// </summary>
    private static string DefaultWeightColumn(List<JoinedSample> samples, string labelColumn)
    {
        bool Taken(string name) =>
            string.Equals(name, labelColumn, StringComparison.OrdinalIgnoreCase)
            || samples.Any(s => s.Input.Keys.Any(k => string.Equals(k, name, StringComparison.OrdinalIgnoreCase)));

        return Taken(RetrainingColumns.DefaultWeight) ? "feedback_weight" : RetrainingColumns.DefaultWeight;
    }

    private static async Task WriteCsvAsync(
        List<JoinedSample> samples,
        string outputPath,
        string labelColumn,
        string? weightColumn,
        CancellationToken cancellationToken)
    {
        if (samples.Count == 0)
            return;

        // The logged inputs are the features. A logged column that shares the label's or the weight's
        // name is that column: the label takes the feedback's value when there is one.
        static bool Same(string a, string? b) => string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
        var columns = samples
            .SelectMany(s => s.Input.Keys)
            .Where(k => !Same(k, labelColumn) && !Same(k, weightColumn))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(c => c, StringComparer.Ordinal)
            .ToList();

        var sb = new StringBuilder();
        var header = columns.Append(labelColumn);
        if (weightColumn is not null)
            header = header.Append(weightColumn);
        sb.AppendLine(CsvFieldParser.FormatLine(header));

        foreach (var sample in samples)
        {
            var values = columns
                .Select(col => FormatValue(sample.Input.FirstOrDefault(kv => Same(kv.Key, col)).Value))
                .ToList();

            var label = sample.Feedback?.ActualValue
                ?? sample.Input.FirstOrDefault(kv => Same(kv.Key, labelColumn)).Value;
            values.Add(FormatValue(label));

            if (weightColumn is not null)
                values.Add((sample.Feedback?.EffectiveWeight ?? 1.0).ToString("R", CultureInfo.InvariantCulture));

            sb.AppendLine(CsvFieldParser.FormatLine(values));
        }

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        await File.WriteAllTextAsync(outputPath, sb.ToString(), cancellationToken).ConfigureAwait(false);
    }

    // Numbers are written invariantly — the file is read back by training, not by a person's locale.
    private static string FormatValue(object? value) => value switch
    {
        null => "",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? ""
    };

    private static Dictionary<string, object> ParseInput(JsonElement element)
    {
        var result = new Dictionary<string, object>();
        if (element.ValueKind != JsonValueKind.Object)
            return result;

        foreach (var prop in element.EnumerateObject())
        {
            result[prop.Name] = GetObjectFromJson(prop.Value) ?? "";
        }
        return result;
    }

    private static object? GetObjectFromJson(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            _ => element.GetRawText()
        };
    }

    private static string SanitizeFileName(string name)
    {
        var invalidChars = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(name.Length);
        foreach (var c in name)
        {
            sb.Append(invalidChars.Contains(c) ? '_' : c);
        }
        return sb.ToString();
    }

    /// <summary>A logged prediction as read for sampling: what the strategies look at, and its line.</summary>
    private sealed record LoggedPrediction(
        string PredictionId,
        object? Output,
        double? Confidence,
        DateTimeOffset Timestamp,
        string Line)
    {
        /// <summary>The output value a stratified sample groups by.</summary>
        public string StratumKey => Output?.ToString() ?? "_null_";

        /// <summary>The row with its inputs parsed and its feedback, if any — done only for a picked row.</summary>
        public JoinedSample Join(Dictionary<string, FeedbackEntry> feedback)
        {
            using var doc = JsonDocument.Parse(Line);
            var input = doc.RootElement.TryGetProperty("input", out var inputEl)
                ? ParseInput(inputEl)
                : new Dictionary<string, object>();
            return new JoinedSample(PredictionId, input, Output, Confidence, Timestamp, feedback.GetValueOrDefault(PredictionId));
        }
    }

    private record JoinedSample(
        string PredictionId,
        Dictionary<string, object> Input,
        object? Output,
        double? Confidence,
        DateTimeOffset Timestamp,
        FeedbackEntry? Feedback);
}
