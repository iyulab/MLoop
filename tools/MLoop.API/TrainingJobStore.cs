using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

/// <summary>
/// The background training jobs of the project this server serves — one JSON file per job under
/// <c>.mloop/jobs/</c>, held in memory for reads and written through on every change.
/// Thread-safe for concurrent access from API endpoints and background tasks.
/// </summary>
/// <remarks>
/// The jobs used to live only in memory, so a restarted server answered 404 for a job a client was
/// still polling while the experiment it trained sat on disk. On load, a job that was running when
/// the previous process stopped is marked failed — its training did not finish — and a queued one
/// stays queued for <see cref="TrainingJobRunner"/> to pick up, since the 202 that accepted it
/// promised to train it.
/// </remarks>
public class TrainingJobStore
{
    /// <summary>Where the jobs live, under the project's <c>.mloop/</c>.</summary>
    public const string JobsDirectoryName = "jobs";

    /// <summary>The status message of a job the previous server process left running.</summary>
    public const string InterruptedMessage = "Interrupted: the server stopped while this job was running. Start it again with POST /train.";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
    };

    private readonly ConcurrentDictionary<string, TrainingJob> _jobs = new();
    private readonly string _directory;
    private readonly ILogger<TrainingJobStore> _logger;

    public TrainingJobStore(MLoop.API.ProjectRootPath projectRoot, ILogger<TrainingJobStore>? logger = null)
    {
        _directory = Path.Combine(projectRoot.Value, ".mloop", JobsDirectoryName);
        _logger = logger ?? NullLogger<TrainingJobStore>.Instance;
        Load();
    }

    public string CreateJob(TrainingJobRequest request)
    {
        var jobId = $"job-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString()[..8]}";
        var job = new TrainingJob
        {
            JobId = jobId,
            ModelName = request.ModelName,
            DataFile = request.DataFile,
            AdditionalDataFiles = request.AdditionalDataFiles ?? [],
            LabelColumn = request.LabelColumn,
            Task = request.Task,
            TimeLimitSeconds = request.TimeLimitSeconds,
            Metric = request.Metric,
            TestSplit = request.TestSplit,
            MaxRows = request.MaxRows,
            Weight = request.Weight,
            Balance = request.Balance,
            Status = JobStatus.Queued,
            CreatedAt = DateTime.UtcNow
        };
        lock (job.SyncRoot)
        {
            Save(job);
        }
        _jobs[jobId] = job;
        return jobId;
    }

    public TrainingJob? GetJob(string jobId) =>
        _jobs.TryGetValue(jobId, out var job) ? job : null;

    public IReadOnlyList<TrainingJob> GetAllJobs() =>
        _jobs.Values.OrderByDescending(j => j.CreatedAt).ToList();

    /// <summary>The jobs waiting to train, oldest first — the order they were accepted in.</summary>
    public IReadOnlyList<TrainingJob> GetQueuedJobs() =>
        _jobs.Values.Where(j => j.Status == JobStatus.Queued).OrderBy(j => j.CreatedAt).ToList();

    public void UpdateStatus(string jobId, JobStatus status, string? message = null)
    {
        if (_jobs.TryGetValue(jobId, out var job))
        {
            lock (job.SyncRoot)
            {
                job.Status = status;
                if (message != null) job.StatusMessage = message;
                if (status is JobStatus.Completed or JobStatus.Failed)
                    job.CompletedAt = DateTime.UtcNow;
                Save(job);
            }
        }
    }

    public void SetResult(string jobId, string experimentId, Dictionary<string, double>? metrics, string? bestTrainer)
    {
        if (_jobs.TryGetValue(jobId, out var job))
        {
            lock (job.SyncRoot)
            {
                job.ExperimentId = experimentId;
                job.Metrics = metrics;
                job.BestTrainer = bestTrainer;
                Save(job);
            }
        }
    }

    private string PathOf(string jobId) => Path.Combine(_directory, jobId + ".json");

    // Called under the job's lock, so two changes to one job cannot interleave their writes.
    // Written beside the target and moved over it, so a reader never sees half a file.
    private void Save(TrainingJob job)
    {
        try
        {
            Directory.CreateDirectory(_directory);
            var path = PathOf(job.JobId);
            var temp = path + ".tmp";
            File.WriteAllText(temp, JsonSerializer.Serialize(job, JsonOptions));
            File.Move(temp, path, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // The job still runs and answers from memory; only surviving a restart is lost.
            _logger.LogWarning(ex, "Could not record training job '{JobId}' to {Directory}", job.JobId, _directory);
        }
    }

    private void Load()
    {
        if (!Directory.Exists(_directory))
            return;

        foreach (var path in Directory.EnumerateFiles(_directory, "*.json"))
        {
            TrainingJob? job;
            try
            {
                job = JsonSerializer.Deserialize<TrainingJob>(File.ReadAllText(path), JsonOptions);
            }
            catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
            {
                _logger.LogWarning(ex, "Skipping unreadable training job record {Path}", path);
                continue;
            }
            if (job is null || Path.GetFileNameWithoutExtension(path) != job.JobId)
            {
                _logger.LogWarning("Skipping training job record {Path}: it does not name its own job", path);
                continue;
            }

            if (job.Status == JobStatus.Running)
            {
                job.Status = JobStatus.Failed;
                job.StatusMessage = InterruptedMessage;
                job.CompletedAt = DateTime.UtcNow;
                Save(job);
            }
            _jobs[job.JobId] = job;
        }
    }
}

public class TrainingJob
{
    [JsonIgnore]
    internal readonly object SyncRoot = new();

    public required string JobId { get; init; }
    public required string ModelName { get; init; }
    public required string DataFile { get; init; }
    public IReadOnlyList<string> AdditionalDataFiles { get; init; } = [];
    public required string LabelColumn { get; init; }
    public required string Task { get; init; }
    public int? TimeLimitSeconds { get; set; }
    public string Metric { get; set; } = "auto";
    public double? TestSplit { get; set; }
    public int? MaxRows { get; set; }
    public string? Weight { get; set; }
    public string? Balance { get; set; }

    public JobStatus Status { get; set; }
    public string? StatusMessage { get; set; }
    public DateTime CreatedAt { get; init; }
    public DateTime? CompletedAt { get; set; }

    // Result fields
    public string? ExperimentId { get; set; }
    public Dictionary<string, double>? Metrics { get; set; }
    public string? BestTrainer { get; set; }
}

/// <summary>A <c>POST /train</c> body.</summary>
/// <param name="DataFile">The training data, within the project.</param>
/// <param name="LabelColumn">The label column (the request's, in the place of <c>--label</c>).</param>
/// <param name="Task">The task (in the place of <c>--task</c>).</param>
/// <param name="Name">The model; <c>default</c> when absent.</param>
/// <param name="Metric">The metric to optimize; <c>auto</c> leaves it to mloop.yaml, then the task.</param>
/// <param name="MaxRows">Rows to sample the data down to, as <c>--max-rows</c>.</param>
/// <param name="TimeLimitSeconds">The training budget; null leaves it to mloop.yaml, then the default — as an absent <c>--time</c> does.</param>
/// <param name="TestSplit">The held-out share; null leaves it to mloop.yaml, then the default.</param>
/// <param name="Weight">The column that weights the rows, as <c>--weight</c> — the <c>weightColumn</c> <c>POST /samples</c> answers with.</param>
/// <param name="AdditionalDataFiles">
/// Files merged after <paramref name="DataFile"/>, as <c>mloop train --data a.csv b.csv</c> — a retraining set from
/// <c>POST /samples</c>, say. A file without the model's weight column counts its rows 1.
/// </param>
/// <param name="Balance">
/// Class balancing, as <c>--balance</c>: <c>auto</c>, <c>none</c> or a target ratio — the train part only, split off
/// from the test part first. Classification only, and not with a weight column.
/// </param>
public record TrainingJobRequest(
    string DataFile,
    string LabelColumn,
    string Task,
    string? Name = null,
    int? TimeLimitSeconds = null,
    string Metric = "auto",
    double? TestSplit = null,
    int? MaxRows = null,
    IReadOnlyList<string>? AdditionalDataFiles = null,
    string? Weight = null,
    string? Balance = null)
{
    public string ModelName => string.IsNullOrWhiteSpace(Name) ? "default" : Name.Trim().ToLowerInvariant();
}

public enum JobStatus
{
    Queued,
    Running,
    Completed,
    Failed
}
