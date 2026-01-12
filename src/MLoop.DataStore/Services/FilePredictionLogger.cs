using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

using MLoop.DataStore.Interfaces;

namespace MLoop.DataStore.Services;

/// <summary>
/// Filesystem-based prediction logger using JSONL format.
/// Stores logs in .mloop/logs/{model}/{date}.jsonl
/// </summary>
public sealed class FilePredictionLogger : IPredictionLogger
{
    private readonly string _baseDirectory;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly object _writeLock = new();

    public FilePredictionLogger(string baseDirectory)
    {
        _baseDirectory = Path.Combine(baseDirectory, ".mloop", "logs");
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter() }
        };
    }

    /// <inheritdoc />
    public async Task LogPredictionAsync(
        string modelName,
        string experimentId,
        IDictionary<string, object> input,
        object output,
        double? confidence = null,
        CancellationToken cancellationToken = default)
    {
        var entry = new PredictionLogEntry(
            modelName,
            experimentId,
            input,
            output,
            confidence,
            DateTimeOffset.UtcNow);

        await WriteEntryAsync(entry, cancellationToken);
    }

    /// <inheritdoc />
    public async Task LogBatchAsync(
        string modelName,
        string experimentId,
        IEnumerable<PredictionLogEntry> entries,
        CancellationToken cancellationToken = default)
    {
        var entriesList = entries.ToList();
        if (entriesList.Count == 0)
            return;

        var sb = new StringBuilder();
        foreach (var entry in entriesList)
        {
            var serializable = ToSerializable(entry);
            sb.AppendLine(JsonSerializer.Serialize(serializable, _jsonOptions));
        }

        var filePath = GetLogFilePath(modelName, DateTimeOffset.UtcNow);
        EnsureDirectoryExists(Path.GetDirectoryName(filePath)!);

        lock (_writeLock)
        {
            File.AppendAllText(filePath, sb.ToString());
        }

        await Task.CompletedTask;
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<PredictionLogEntry>> GetLogsAsync(
        string? modelName = null,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        int limit = 1000,
        CancellationToken cancellationToken = default)
    {
        var results = new List<PredictionLogEntry>();
        var effectiveFrom = from ?? DateTimeOffset.MinValue;
        var effectiveTo = to ?? DateTimeOffset.MaxValue;

        var directories = GetModelDirectories(modelName);

        foreach (var dir in directories)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            var logFiles = Directory.GetFiles(dir, "*.jsonl")
                .Where(f => IsFileInDateRange(f, effectiveFrom, effectiveTo))
                .OrderByDescending(f => f);

            foreach (var file in logFiles)
            {
                if (cancellationToken.IsCancellationRequested || results.Count >= limit)
                    break;

                var entries = await ReadEntriesFromFileAsync(file, effectiveFrom, effectiveTo, cancellationToken);
                results.AddRange(entries);

                if (results.Count >= limit)
                    break;
            }
        }

        return results
            .OrderByDescending(e => e.Timestamp)
            .Take(limit)
            .ToList();
    }

    private async Task WriteEntryAsync(PredictionLogEntry entry, CancellationToken cancellationToken)
    {
        var serializable = ToSerializable(entry);
        var json = JsonSerializer.Serialize(serializable, _jsonOptions);
        var filePath = GetLogFilePath(entry.ModelName, entry.Timestamp);

        EnsureDirectoryExists(Path.GetDirectoryName(filePath)!);

        lock (_writeLock)
        {
            File.AppendAllText(filePath, json + Environment.NewLine);
        }

        await Task.CompletedTask;
    }

    private async Task<IEnumerable<PredictionLogEntry>> ReadEntriesFromFileAsync(
        string filePath,
        DateTimeOffset from,
        DateTimeOffset to,
        CancellationToken cancellationToken)
    {
        var entries = new List<PredictionLogEntry>();

        try
        {
            var lines = await File.ReadAllLinesAsync(filePath, cancellationToken);
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                try
                {
                    var serializable = JsonSerializer.Deserialize<SerializablePredictionLog>(line, _jsonOptions);
                    if (serializable != null)
                    {
                        var entry = FromSerializable(serializable);
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

    private string GetLogFilePath(string modelName, DateTimeOffset timestamp)
    {
        var safeModelName = SanitizeFileName(modelName);
        var dateStr = timestamp.UtcDateTime.ToString("yyyy-MM-dd");
        return Path.Combine(_baseDirectory, safeModelName, $"{dateStr}.jsonl");
    }

    private IEnumerable<string> GetModelDirectories(string? modelName)
    {
        if (!Directory.Exists(_baseDirectory))
            return [];

        if (modelName != null)
        {
            var safeModelName = SanitizeFileName(modelName);
            var modelDir = Path.Combine(_baseDirectory, safeModelName);
            return Directory.Exists(modelDir) ? [modelDir] : [];
        }

        return Directory.GetDirectories(_baseDirectory);
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

    private static SerializablePredictionLog ToSerializable(PredictionLogEntry entry)
    {
        return new SerializablePredictionLog
        {
            Id = Guid.NewGuid().ToString("N")[..16],
            ModelName = entry.ModelName,
            ExperimentId = entry.ExperimentId,
            Input = entry.Input.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
            Output = entry.Output,
            Confidence = entry.Confidence,
            Timestamp = entry.Timestamp
        };
    }

    private static PredictionLogEntry FromSerializable(SerializablePredictionLog log)
    {
        return new PredictionLogEntry(
            log.ModelName ?? string.Empty,
            log.ExperimentId ?? string.Empty,
            log.Input ?? new Dictionary<string, object>(),
            log.Output ?? new object(),
            log.Confidence,
            log.Timestamp);
    }

    /// <summary>
    /// Internal serializable format with prediction ID.
    /// </summary>
    private sealed class SerializablePredictionLog
    {
        public string? Id { get; set; }
        public string? ModelName { get; set; }
        public string? ExperimentId { get; set; }
        public Dictionary<string, object>? Input { get; set; }
        public object? Output { get; set; }
        public double? Confidence { get; set; }
        public DateTimeOffset Timestamp { get; set; }
    }
}
