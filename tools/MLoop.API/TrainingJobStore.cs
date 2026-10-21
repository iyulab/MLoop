using System.Collections.Concurrent;

/// <summary>
/// In-memory store for background training jobs.
/// Thread-safe for concurrent access from API endpoints and background tasks.
/// </summary>
public class TrainingJobStore
{
    private readonly ConcurrentDictionary<string, TrainingJob> _jobs = new();

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
            Status = JobStatus.Queued,
            CreatedAt = DateTime.UtcNow
        };
        _jobs[jobId] = job;
        return jobId;
    }

    public TrainingJob? GetJob(string jobId) =>
        _jobs.TryGetValue(jobId, out var job) ? job : null;

    public IReadOnlyList<TrainingJob> GetAllJobs() =>
        _jobs.Values.OrderByDescending(j => j.CreatedAt).ToList();

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
            }
        }
    }
}

public class TrainingJob
{
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
/// <param name="AdditionalDataFiles">
/// Files merged after <paramref name="DataFile"/>, as <c>mloop train --data a.csv b.csv</c> — a retraining set from
/// <c>POST /samples</c>, say. A file without the model's weight column counts its rows 1.
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
    IReadOnlyList<string>? AdditionalDataFiles = null)
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
