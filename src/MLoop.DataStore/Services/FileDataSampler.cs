using System.Text;
using System.Text.Json;
using MLoop.DataStore.Interfaces;

namespace MLoop.DataStore.Services;

/// <summary>
/// Filesystem-based data sampler that creates retraining datasets from predictions and feedback.
/// Supports Random, Recent, and FeedbackPriority strategies.
/// </summary>
public sealed class FileDataSampler : IDataSampler
{
    private readonly string _logsDirectory;
    private readonly string _feedbackDirectory;
    private readonly JsonSerializerOptions _jsonOptions;

    public FileDataSampler(string baseDirectory)
    {
        _logsDirectory = Path.Combine(baseDirectory, ".mloop", "logs");
        _feedbackDirectory = Path.Combine(baseDirectory, ".mloop", "feedback");
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
        CancellationToken cancellationToken = default)
    {
        var predictions = await LoadPredictionsAsync(modelName, cancellationToken);
        var feedback = await LoadFeedbackAsync(modelName, cancellationToken);

        if (predictions.Count == 0)
        {
            throw new InvalidOperationException(
                $"No predictions found for model '{modelName}'. " +
                "Log predictions using 'mloop predict --log' first.");
        }

        // Join predictions with feedback
        var feedbackDict = feedback.ToDictionary(f => f.PredictionId, f => f.ActualValue);
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

        // Write to CSV
        await WriteCsvAsync(sampled, outputPath, cancellationToken);

        return new SamplingResult(
            OutputPath: outputPath,
            SampledCount: sampled.Count,
            TotalAvailable: predictions.Count,
            StrategyUsed: strategy,
            CreatedAt: DateTimeOffset.UtcNow);
    }

    /// <inheritdoc />
    public async Task<SamplingStatistics> GetStatisticsAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var predictions = await LoadPredictionsAsync(modelName, cancellationToken);
        var feedback = await LoadFeedbackAsync(modelName, cancellationToken);

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
            SamplingStrategy.Stratified => RandomSample(data, actualSize), // Fallback to random
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
        var withFeedback = data.Where(d => d.ActualValue != null).ToList();
        var withoutFeedback = data.Where(d => d.ActualValue == null).ToList();

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
                var lines = await File.ReadAllLinesAsync(file, cancellationToken);
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

    private async Task<List<FeedbackRecord>> LoadFeedbackAsync(
        string modelName,
        CancellationToken cancellationToken)
    {
        var feedback = new List<FeedbackRecord>();
        var modelDir = Path.Combine(_feedbackDirectory, SanitizeFileName(modelName));

        if (!Directory.Exists(modelDir))
            return feedback;

        var feedbackFiles = Directory.GetFiles(modelDir, "*.jsonl");
        foreach (var file in feedbackFiles)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

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

                        var predictionId = root.TryGetProperty("predictionId", out var pidEl)
                            ? pidEl.GetString()
                            : null;

                        var actualValue = root.TryGetProperty("actualValue", out var avEl)
                            ? GetObjectFromJson(avEl)
                            : null;

                        if (predictionId != null && actualValue != null)
                        {
                            feedback.Add(new FeedbackRecord(predictionId, actualValue));
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
                // Skip inaccessible files
            }
        }

        return feedback;
    }

    private async Task WriteCsvAsync(
        List<JoinedSample> samples,
        string outputPath,
        CancellationToken cancellationToken)
    {
        if (samples.Count == 0)
            return;

        // Collect all unique input columns
        var allColumns = new HashSet<string>();
        foreach (var sample in samples)
        {
            foreach (var key in sample.Input.Keys)
            {
                allColumns.Add(key);
            }
        }

        var columns = allColumns.OrderBy(c => c).ToList();
        var sb = new StringBuilder();

        // Header: input columns + Output + ActualValue (if any have feedback)
        var hasFeedback = samples.Any(s => s.ActualValue != null);
        sb.Append(string.Join(",", columns.Select(EscapeCsvField)));
        sb.Append(",Output");
        if (hasFeedback)
        {
            sb.Append(",ActualValue");
        }
        sb.AppendLine();

        // Data rows
        foreach (var sample in samples)
        {
            var values = new List<string>();
            foreach (var col in columns)
            {
                var value = sample.Input.TryGetValue(col, out var v) ? v?.ToString() ?? "" : "";
                values.Add(EscapeCsvField(value));
            }
            values.Add(EscapeCsvField(sample.Output?.ToString() ?? ""));
            if (hasFeedback)
            {
                values.Add(EscapeCsvField(sample.ActualValue?.ToString() ?? ""));
            }
            sb.AppendLine(string.Join(",", values));
        }

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        await File.WriteAllTextAsync(outputPath, sb.ToString(), cancellationToken);
    }

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

    private static string EscapeCsvField(string field)
    {
        if (field.Contains(',') || field.Contains('"') || field.Contains('\n'))
        {
            return $"\"{field.Replace("\"", "\"\"")}\"";
        }
        return field;
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

    private record FeedbackRecord(string PredictionId, object ActualValue);

    private record JoinedSample(
        string PredictionId,
        Dictionary<string, object> Input,
        object? Output,
        double? Confidence,
        DateTimeOffset Timestamp,
        object? ActualValue);
}
