using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using MLoop.Core.Storage;
using MLoop.DataStore.Interfaces;

namespace MLoop.DataStore.Services;

/// <summary>
/// Filesystem-based feedback collector storing feedback in JSONL format.
/// Links feedback to predictions for model monitoring and retraining decisions.
/// </summary>
public sealed class FileFeedbackCollector : IFeedbackCollector
{
    private readonly string _feedbackDirectory;
    private readonly string _logsDirectory;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly object _writeLock = new();

    public FileFeedbackCollector(string baseDirectory)
    {
        _feedbackDirectory = Path.Combine(baseDirectory, ".mloop", "feedback");
        _logsDirectory = Path.Combine(baseDirectory, ".mloop", "logs");
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter() }
        };
    }

    /// <inheritdoc />
    public async Task RecordFeedbackAsync(
        string predictionId,
        object actualValue,
        string? source = null,
        double? weight = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(predictionId);
        ArgumentNullException.ThrowIfNull(actualValue);
        FeedbackEntry.ValidateWeight(weight);

        // Find the original prediction to get modelName and predictedValue
        var prediction = await FindPredictionByIdAsync(predictionId, cancellationToken).ConfigureAwait(false);
        if (prediction == null)
            throw new PredictionNotFoundException(predictionId);


        var entry = new SerializableFeedbackEntry
        {
            PredictionId = predictionId,
            ModelName = prediction.Value.ModelName,
            ExperimentId = prediction.Value.ExperimentId,
            PredictedValue = prediction.Value.Output,
            ActualValue = actualValue,
            Source = source,
            Weight = weight,
            Timestamp = DateTimeOffset.UtcNow
        };

        await WriteEntryAsync(entry, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<FeedbackEntry>> GetFeedbackAsync(
        string modelName,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        int limit = 1000,
        CancellationToken cancellationToken = default)
    {
        var effectiveFrom = from ?? DateTimeOffset.MinValue;
        var effectiveTo = to ?? DateTimeOffset.MaxValue;

        var modelDir = Path.Combine(_feedbackDirectory, SanitizeFileName(modelName));
        if (!Directory.Exists(modelDir))
            return [];

        var newest = new NewestEntries<FeedbackEntry>(limit, e => e.Timestamp);
        foreach (var file in JsonLines.DayFilesNewestFirst(modelDir, effectiveFrom, effectiveTo))
        {
            if (cancellationToken.IsCancellationRequested)
                break;
            if (JsonLines.EndOfDay(file) is { } end && newest.HoldsAllNewerThan(end))
                break;

            await foreach (var entry in ReadEntriesFromFileAsync(file, effectiveFrom, effectiveTo, cancellationToken).ConfigureAwait(false))
                newest.Add(entry);
        }

        return newest.NewestFirst();
    }

    /// <inheritdoc />
    public async Task<FeedbackMetrics> CalculateMetricsAsync(
        string modelName,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        CancellationToken cancellationToken = default,
        string? experimentId = null)
    {
        var feedback = await GetFeedbackAsync(modelName, from, to, int.MaxValue, cancellationToken).ConfigureAwait(false);
        // One model's accuracy: the answers of the experiment named, not of whatever answered under the model name
        // before a promotion or as a shadow candidate. Feedback recorded before it named its experiment is read
        // against its prediction's log line; what no log line answers any more is left out — and counted.
        var unattributed = 0;
        if (experimentId is not null)
        {
            feedback = await AttributeAsync(modelName, feedback, cancellationToken).ConfigureAwait(false);
            unattributed = feedback.Count(e => e.ExperimentId is null);
            feedback = [.. feedback.Where(e => string.Equals(e.ExperimentId, experimentId, StringComparison.Ordinal))];
        }

        if (feedback.Count == 0)
        {
            return new FeedbackMetrics(
                modelName,
                TotalFeedback: 0,
                Accuracy: null,
                CalculatedAt: DateTimeOffset.UtcNow,
                ExperimentId: experimentId,
                UnattributedFeedback: unattributed);
        }

        // Recording feedback again for a prediction corrects it — the latest actual value stands, as
        // sampling reads it too — so each prediction is counted once.
        feedback = FeedbackEntry.LatestPerPrediction(feedback);

        // Each prediction counts by its weight: a sample reviewed with unequal probabilities misstates
        // accuracy when its rows are counted alike. Without weights every row counts 1 — plain accuracy.
        var weighted = feedback.Any(e => e.Weight is not null);
        double total = 0, correct = 0, squares = 0;
        foreach (var entry in feedback)
        {
            total += entry.EffectiveWeight;
            squares += entry.EffectiveWeight * entry.EffectiveWeight;
            if (FeedbackValueComparer.ValuesMatch(entry.PredictedValue, entry.ActualValue))
                correct += entry.EffectiveWeight;
        }

        double? accuracy = total > 0 ? correct / total : null;
        double? effectiveSize = squares > 0 ? total * total / squares : null;

        return new FeedbackMetrics(
            modelName,
            TotalFeedback: feedback.Count,
            Accuracy: accuracy,
            CalculatedAt: DateTimeOffset.UtcNow,
            Weighted: weighted,
            EffectiveSampleSize: effectiveSize,
            ExperimentId: experimentId,
            UnattributedFeedback: unattributed);
    }

    private async Task WriteEntryAsync(SerializableFeedbackEntry entry, CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(entry, _jsonOptions);
        var filePath = GetFeedbackFilePath(entry.ModelName!, entry.Timestamp);

        EnsureDirectoryExists(Path.GetDirectoryName(filePath)!);

        lock (_writeLock)
        {
            File.AppendAllText(filePath, json + Environment.NewLine);
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Names the experiment of feedback that does not — written before feedback recorded it — from its prediction's
    /// log line, reading <paramref name="modelName"/>'s logs once and only when such feedback exists.
    /// </summary>
    private async Task<IReadOnlyList<FeedbackEntry>> AttributeAsync(
        string modelName, IReadOnlyList<FeedbackEntry> feedback, CancellationToken cancellationToken)
    {
        var wanted = feedback.Where(e => e.ExperimentId is null).Select(e => e.PredictionId).ToHashSet(StringComparer.Ordinal);
        var modelLogs = Path.Combine(_logsDirectory, SanitizeFileName(modelName));
        if (wanted.Count == 0 || !Directory.Exists(modelLogs))
            return feedback;

        var found = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(modelLogs, "*.jsonl"))
        {
            await foreach (var line in JsonLines.ReadAsync(file, cancellationToken).ConfigureAwait(false))
            {
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    if (root.TryGetProperty("id", out var id) && id.GetString() is { } predictionId && wanted.Contains(predictionId)
                        && root.TryGetProperty("experimentId", out var experiment) && experiment.ValueKind == JsonValueKind.String
                        && experiment.GetString() is { Length: > 0 } answered)
                        found[predictionId] = answered;
                }
                catch (JsonException)
                {
                    // A torn line names nothing.
                }
            }
            if (found.Count == wanted.Count)
                break;
        }

        return found.Count == 0
            ? feedback
            : [.. feedback.Select(e => e.ExperimentId is null && found.TryGetValue(e.PredictionId, out var answered) ? e with { ExperimentId = answered } : e)];
    }

    private async Task<(string ModelName, string? ExperimentId, object Output)?> FindPredictionByIdAsync(
        string predictionId,
        CancellationToken cancellationToken)
    {
        if (!Directory.Exists(_logsDirectory))
            return null;

        // Every logged day, newest first across models: the prediction feedback answers is usually recent, so
        // the search ends early, and one older than any fixed window is still found. A window of 7 days used to
        // stand here — feedback that arrived later was refused as "prediction not found" for a prediction that
        // was on disk. How far back logs reach is the project's retention period (mloop logs prune).
        var logFiles = Directory.GetDirectories(_logsDirectory)
            .SelectMany(modelDir => Directory.GetFiles(modelDir, "*.jsonl").Select(file => (ModelName: Path.GetFileName(modelDir), File: file)))
            .OrderByDescending(f => Path.GetFileName(f.File), StringComparer.Ordinal);

        // An id the serializer writes as it is (the store's own ids are hex) appears in its line verbatim, so a line
        // without it is passed over unparsed — a day of a busy model is tens of thousands of lines. An id with a
        // character the serializer escapes is looked for by parsing every line.
        var idIsVerbatim = predictionId.Length > 0 && predictionId.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.' or ':');

        foreach (var (modelName, file) in logFiles)
        {
            if (cancellationToken.IsCancellationRequested)
                return null;

            await foreach (var line in JsonLines.ReadAsync(file, cancellationToken).ConfigureAwait(false))
            {
                // Every line of a day is a candidate; only the ones that can name this id are parsed.
                if (idIsVerbatim && !line.Contains(predictionId, StringComparison.Ordinal))
                    continue;

                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    if (root.TryGetProperty("id", out var idElement) &&
                        idElement.GetString() == predictionId)
                    {
                        var output = root.TryGetProperty("output", out var outputElement)
                            ? GetObjectFromJsonElement(outputElement)
                            : new object();

                        var experimentId = root.TryGetProperty("experimentId", out var experimentElement)
                                           && experimentElement.ValueKind == JsonValueKind.String
                                           && experimentElement.GetString() is { Length: > 0 } answered
                            ? answered
                            : null;

                        return (modelName, experimentId, output);
                    }
                }
                catch (JsonException)
                {
                    // Skip malformed entries
                }
            }
        }

        return null;
    }

    private async IAsyncEnumerable<FeedbackEntry> ReadEntriesFromFileAsync(
        string filePath,
        DateTimeOffset from,
        DateTimeOffset to,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var line in JsonLines.ReadAsync(filePath, cancellationToken).ConfigureAwait(false))
        {
            SerializableFeedbackEntry? serializable;
            try
            {
                serializable = JsonSerializer.Deserialize<SerializableFeedbackEntry>(line, _jsonOptions);
            }
            catch (JsonException)
            {
                continue; // A malformed or torn line is skipped.
            }

            if (serializable is null)
                continue;

            var entry = new FeedbackEntry(
                serializable.PredictionId ?? string.Empty,
                serializable.ModelName ?? string.Empty,
                serializable.PredictedValue ?? new object(),
                serializable.ActualValue ?? new object(),
                serializable.Source,
                serializable.Timestamp,
                serializable.Weight,
                serializable.ExperimentId);

            if (entry.Timestamp >= from && entry.Timestamp <= to)
                yield return entry;
        }
    }

    private string GetFeedbackFilePath(string modelName, DateTimeOffset timestamp)
    {
        var safeModelName = SanitizeFileName(modelName);
        return Path.Combine(_feedbackDirectory, safeModelName, $"{TimestampKeys.Day(timestamp)}.jsonl");
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

    private static void EnsureDirectoryExists(string path)
    {
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
    }

    private static object GetObjectFromJsonElement(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.String => element.GetString() ?? string.Empty,
            JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => new object(),
            _ => element.GetRawText()
        };
    }

    /// <summary>
    /// Internal serializable format for feedback entries.
    /// </summary>
    private sealed class SerializableFeedbackEntry
    {
        public string? PredictionId { get; set; }
        public string? ModelName { get; set; }
        public string? ExperimentId { get; set; }
        public object? PredictedValue { get; set; }
        public object? ActualValue { get; set; }
        public string? Source { get; set; }
        public double? Weight { get; set; }
        public DateTimeOffset Timestamp { get; set; }
    }
}
