using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
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
    private const int DefaultSearchDays = 7;

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
        CancellationToken cancellationToken = default)
    {
        // Find the original prediction to get modelName and predictedValue
        var prediction = await FindPredictionByIdAsync(predictionId, cancellationToken);
        if (prediction == null)
        {
            throw new InvalidOperationException(
                $"Prediction with ID '{predictionId}' not found in logs. " +
                "Ensure the prediction was logged and try again.");
        }

        var entry = new SerializableFeedbackEntry
        {
            PredictionId = predictionId,
            ModelName = prediction.Value.ModelName,
            PredictedValue = prediction.Value.Output,
            ActualValue = actualValue,
            Source = source,
            Timestamp = DateTimeOffset.UtcNow
        };

        await WriteEntryAsync(entry, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<FeedbackEntry>> GetFeedbackAsync(
        string modelName,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        int limit = 1000,
        CancellationToken cancellationToken = default)
    {
        var results = new List<FeedbackEntry>();
        var effectiveFrom = from ?? DateTimeOffset.MinValue;
        var effectiveTo = to ?? DateTimeOffset.MaxValue;

        var modelDir = Path.Combine(_feedbackDirectory, SanitizeFileName(modelName));
        if (!Directory.Exists(modelDir))
            return results;

        var feedbackFiles = Directory.GetFiles(modelDir, "*.jsonl")
            .Where(f => IsFileInDateRange(f, effectiveFrom, effectiveTo))
            .OrderByDescending(f => f);

        foreach (var file in feedbackFiles)
        {
            if (cancellationToken.IsCancellationRequested || results.Count >= limit)
                break;

            var entries = await ReadEntriesFromFileAsync(file, effectiveFrom, effectiveTo, cancellationToken);
            results.AddRange(entries);

            if (results.Count >= limit)
                break;
        }

        return results
            .OrderByDescending(e => e.Timestamp)
            .Take(limit)
            .ToList();
    }

    /// <inheritdoc />
    public async Task<FeedbackMetrics> CalculateMetricsAsync(
        string modelName,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        CancellationToken cancellationToken = default)
    {
        var feedback = await GetFeedbackAsync(modelName, from, to, int.MaxValue, cancellationToken);

        if (feedback.Count == 0)
        {
            return new FeedbackMetrics(
                modelName,
                TotalPredictions: 0,
                TotalFeedback: 0,
                Accuracy: null,
                Precision: null,
                Recall: null,
                CalculatedAt: DateTimeOffset.UtcNow);
        }

        // Calculate accuracy for classification tasks
        int correctCount = 0;
        foreach (var entry in feedback)
        {
            if (ValuesMatch(entry.PredictedValue, entry.ActualValue))
            {
                correctCount++;
            }
        }

        var accuracy = (double)correctCount / feedback.Count;

        return new FeedbackMetrics(
            modelName,
            TotalPredictions: feedback.Count,
            TotalFeedback: feedback.Count,
            Accuracy: accuracy,
            Precision: null, // Would require class-specific calculation
            Recall: null,    // Would require class-specific calculation
            CalculatedAt: DateTimeOffset.UtcNow);
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

    private async Task<(string ModelName, object Output)?> FindPredictionByIdAsync(
        string predictionId,
        CancellationToken cancellationToken)
    {
        if (!Directory.Exists(_logsDirectory))
            return null;

        var modelDirs = Directory.GetDirectories(_logsDirectory);
        var searchFrom = DateTimeOffset.UtcNow.AddDays(-DefaultSearchDays);

        foreach (var modelDir in modelDirs)
        {
            var modelName = Path.GetFileName(modelDir);
            var logFiles = Directory.GetFiles(modelDir, "*.jsonl")
                .Where(f => IsFileInDateRange(f, searchFrom, DateTimeOffset.MaxValue))
                .OrderByDescending(f => f);

            foreach (var file in logFiles)
            {
                if (cancellationToken.IsCancellationRequested)
                    return null;

                try
                {
                    var lines = await File.ReadAllLinesAsync(file, cancellationToken);
                    foreach (var line in lines)
                    {
                        if (string.IsNullOrWhiteSpace(line))
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

                                return (modelName, output);
                            }
                        }
                        catch (JsonException)
                        {
                            // Skip malformed entries
                        }
                    }
                }
                catch (IOException)
                {
                    // File might be locked or inaccessible
                }
            }
        }

        return null;
    }

    private async Task<IEnumerable<FeedbackEntry>> ReadEntriesFromFileAsync(
        string filePath,
        DateTimeOffset from,
        DateTimeOffset to,
        CancellationToken cancellationToken)
    {
        var entries = new List<FeedbackEntry>();

        try
        {
            var lines = await File.ReadAllLinesAsync(filePath, cancellationToken);
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                try
                {
                    var serializable = JsonSerializer.Deserialize<SerializableFeedbackEntry>(line, _jsonOptions);
                    if (serializable != null)
                    {
                        var entry = new FeedbackEntry(
                            serializable.PredictionId ?? string.Empty,
                            serializable.ModelName ?? string.Empty,
                            serializable.PredictedValue ?? new object(),
                            serializable.ActualValue ?? new object(),
                            serializable.Source,
                            serializable.Timestamp);

                        if (entry.Timestamp >= from && entry.Timestamp <= to)
                        {
                            entries.Add(entry);
                        }
                    }
                }
                catch (JsonException)
                {
                    // Skip malformed entries
                }
            }
        }
        catch (IOException)
        {
            // File might be locked or inaccessible
        }

        return entries;
    }

    private string GetFeedbackFilePath(string modelName, DateTimeOffset timestamp)
    {
        var safeModelName = SanitizeFileName(modelName);
        var dateStr = timestamp.UtcDateTime.ToString("yyyy-MM-dd");
        return Path.Combine(_feedbackDirectory, safeModelName, $"{dateStr}.jsonl");
    }

    private static bool IsFileInDateRange(string filePath, DateTimeOffset from, DateTimeOffset to)
    {
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        if (DateOnly.TryParseExact(fileName, "yyyy-MM-dd", null, System.Globalization.DateTimeStyles.None, out var fileDate))
        {
            var fileDateTime = new DateTimeOffset(fileDate.ToDateTime(TimeOnly.MinValue), TimeSpan.Zero);
            var nextDay = fileDateTime.AddDays(1);
            return fileDateTime <= to && nextDay > from;
        }
        return true; // Include if we can't parse the date
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

    private static bool ValuesMatch(object predicted, object actual)
    {
        if (predicted == null && actual == null)
            return true;
        if (predicted == null || actual == null)
            return false;

        // Handle JsonElement comparisons
        if (predicted is JsonElement predictedJson)
            predicted = GetObjectFromJsonElement(predictedJson);
        if (actual is JsonElement actualJson)
            actual = GetObjectFromJsonElement(actualJson);

        // String comparison (case-insensitive for labels)
        if (predicted is string predictedStr && actual is string actualStr)
            return string.Equals(predictedStr, actualStr, StringComparison.OrdinalIgnoreCase);

        // Numeric comparison with tolerance
        if (IsNumeric(predicted) && IsNumeric(actual))
        {
            var predictedNum = Convert.ToDouble(predicted);
            var actualNum = Convert.ToDouble(actual);
            return Math.Abs(predictedNum - actualNum) < 0.0001;
        }

        return predicted.Equals(actual);
    }

    private static bool IsNumeric(object? value)
    {
        return value is byte or sbyte or short or ushort or int or uint
            or long or ulong or float or double or decimal;
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
        public object? PredictedValue { get; set; }
        public object? ActualValue { get; set; }
        public string? Source { get; set; }
        public DateTimeOffset Timestamp { get; set; }
    }
}
