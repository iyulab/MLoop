using System.Globalization;
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
    private readonly JsonSerializerOptions _jsonOptions;

    public FileDataSampler(string baseDirectory)
    {
        _logsDirectory = Path.Combine(baseDirectory, ".mloop", "logs");
        // Feedback is read through its own store — one reader of the format, so a field it gains
        // (a weight) reaches sampling without a second parser to update.
        _feedback = new FileFeedbackCollector(baseDirectory);
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
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

        var predictions = await LoadPredictionsAsync(modelName, cancellationToken).ConfigureAwait(false);
        var feedback = await LoadFeedbackAsync(modelName, cancellationToken).ConfigureAwait(false);

        if (predictions.Count == 0)
        {
            throw new InvalidOperationException(
                $"No predictions found for model '{modelName}'. " +
                "Log predictions using 'mloop predict --log' first.");
        }

        // Join predictions with feedback. Feedback can be recorded more than once for the same
        // prediction (RecordFeedbackAsync appends — e.g. a user correcting a label): the latest stands.
        var feedbackDict = FeedbackEntry.LatestPerPrediction(feedback)
            .ToDictionary(f => f.PredictionId);
        var joinedData = predictions
            .Select(p => new JoinedSample(
                p.PredictionId,
                p.Input,
                p.Output,
                p.Confidence,
                p.Timestamp,
                feedbackDict.GetValueOrDefault(p.PredictionId)))
            .ToList();

        // Apply sampling strategy
        var sampled = ApplyStrategy(joinedData, sampleSize, strategy);

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
            TotalAvailable: predictions.Count,
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
        var predictions = await LoadPredictionsAsync(modelName, cancellationToken).ConfigureAwait(false);
        var feedback = await LoadFeedbackAsync(modelName, cancellationToken).ConfigureAwait(false);

        if (predictions.Count == 0)
        {
            return new SamplingStatistics(
                ModelName: modelName,
                TotalPredictions: 0,
                PredictionsWithFeedback: 0,
                LowConfidenceCount: 0,
                OldestEntry: DateTimeOffset.MinValue,
                NewestEntry: DateTimeOffset.MinValue);
        }

        var feedbackIds = new HashSet<string>(feedback.Select(f => f.PredictionId));
        var withFeedback = predictions.Count(p => feedbackIds.Contains(p.PredictionId));
        var lowConfidence = predictions.Count(p => p.Confidence.HasValue && p.Confidence.Value < 0.7);

        return new SamplingStatistics(
            ModelName: modelName,
            TotalPredictions: predictions.Count,
            PredictionsWithFeedback: withFeedback,
            LowConfidenceCount: lowConfidence,
            OldestEntry: predictions.Min(p => p.Timestamp),
            NewestEntry: predictions.Max(p => p.Timestamp));
    }

    private List<JoinedSample> ApplyStrategy(
        List<JoinedSample> data,
        int sampleSize,
        SamplingStrategy strategy)
    {
        var actualSize = Math.Min(sampleSize, data.Count);

        return strategy switch
        {
            SamplingStrategy.Random => RandomSample(data, actualSize),
            SamplingStrategy.Recent => RecentSample(data, actualSize),
            SamplingStrategy.FeedbackPriority => FeedbackPrioritySample(data, actualSize),
            SamplingStrategy.Stratified => StratifiedSample(data, actualSize),
            SamplingStrategy.LowConfidence => LowConfidenceSample(data, actualSize),
            _ => RandomSample(data, actualSize)
        };
    }

    private static List<JoinedSample> RandomSample(List<JoinedSample> data, int size)
    {
        var random = new Random();
        return data.OrderBy(_ => random.Next()).Take(size).ToList();
    }

    private static List<JoinedSample> RecentSample(List<JoinedSample> data, int size)
    {
        return data.OrderByDescending(d => d.Timestamp).Take(size).ToList();
    }

    private static List<JoinedSample> FeedbackPrioritySample(List<JoinedSample> data, int size)
    {
        // Prioritize samples with feedback, then fill with random
        var withFeedback = data.Where(d => d.Feedback != null).ToList();
        var withoutFeedback = data.Where(d => d.Feedback == null).ToList();

        var result = new List<JoinedSample>();
        result.AddRange(withFeedback.Take(size));

        if (result.Count < size)
        {
            var random = new Random();
            var remaining = size - result.Count;
            result.AddRange(withoutFeedback.OrderBy(_ => random.Next()).Take(remaining));
        }

        return result;
    }

    private static List<JoinedSample> LowConfidenceSample(List<JoinedSample> data, int size)
    {
        return data
            .Where(d => d.Confidence.HasValue)
            .OrderBy(d => d.Confidence!.Value)
            .Take(size)
            .ToList();
    }

    private static List<JoinedSample> StratifiedSample(List<JoinedSample> data, int size)
    {
        // Group by output value to maintain class distribution
        var groups = data
            .GroupBy(d => d.Output?.ToString() ?? "_null_")
            .ToDictionary(g => g.Key, g => g.ToList());

        if (groups.Count == 0)
            return new List<JoinedSample>();

        var result = new List<JoinedSample>();
        var random = new Random();

        // Calculate proportional sample size for each group
        var totalCount = data.Count;
        var remaining = size;

        foreach (var (key, group) in groups.OrderByDescending(g => g.Value.Count))
        {
            // Calculate proportional count for this group
            var proportion = (double)group.Count / totalCount;
            var groupSampleSize = Math.Max(1, (int)Math.Round(size * proportion));
            groupSampleSize = Math.Min(groupSampleSize, remaining);
            groupSampleSize = Math.Min(groupSampleSize, group.Count);

            if (groupSampleSize > 0)
            {
                var sampled = group.OrderBy(_ => random.Next()).Take(groupSampleSize);
                result.AddRange(sampled);
                remaining -= groupSampleSize;
            }

            if (remaining <= 0)
                break;
        }

        // Fill remaining slots with random samples if needed
        if (remaining > 0 && result.Count < size)
        {
            var usedIds = new HashSet<string>(result.Select(r => r.PredictionId));
            var unused = data.Where(d => !usedIds.Contains(d.PredictionId)).ToList();
            var additional = unused.OrderBy(_ => random.Next()).Take(remaining);
            result.AddRange(additional);
        }

        return result;
    }

    private async Task<List<PredictionRecord>> LoadPredictionsAsync(
        string modelName,
        CancellationToken cancellationToken)
    {
        var predictions = new List<PredictionRecord>();
        var modelDir = Path.Combine(_logsDirectory, SanitizeFileName(modelName));

        if (!Directory.Exists(modelDir))
            return predictions;

        var logFiles = Directory.GetFiles(modelDir, "*.jsonl");
        foreach (var file in logFiles)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            try
            {
                var lines = await File.ReadAllLinesAsync(file, cancellationToken).ConfigureAwait(false);
                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    try
                    {
                        using var doc = JsonDocument.Parse(line);
                        var root = doc.RootElement;

                        var id = root.TryGetProperty("id", out var idEl) ? idEl.GetString() : null;
                        if (id == null) continue;

                        var input = root.TryGetProperty("input", out var inputEl)
                            ? ParseInput(inputEl)
                            : new Dictionary<string, object>();

                        var output = root.TryGetProperty("output", out var outputEl)
                            ? GetObjectFromJson(outputEl)
                            : null;

                        var confidence = root.TryGetProperty("confidence", out var confEl) && confEl.ValueKind == JsonValueKind.Number
                            ? confEl.GetDouble()
                            : (double?)null;

                        var timestamp = root.TryGetProperty("timestamp", out var tsEl)
                            ? tsEl.GetDateTimeOffset()
                            : DateTimeOffset.MinValue;

                        predictions.Add(new PredictionRecord(id, input, output, confidence, timestamp));
                    }
                    catch (JsonException)
                    {
                        // Skip malformed entries
                    }
                }
            }
            catch (IOException)
            {
                // Skip inaccessible files
            }
        }

        return predictions;
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

    private record PredictionRecord(
        string PredictionId,
        Dictionary<string, object> Input,
        object? Output,
        double? Confidence,
        DateTimeOffset Timestamp);

    private record JoinedSample(
        string PredictionId,
        Dictionary<string, object> Input,
        object? Output,
        double? Confidence,
        DateTimeOffset Timestamp,
        FeedbackEntry? Feedback);
}
