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
    public required string LabelColumn { get; init; }
    public required string Task { get; init; }
    public int TimeLimitSeconds { get; set; }
    public string Metric { get; set; } = "auto";
    public double TestSplit { get; set; } = 0.2;
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

public record TrainingJobRequest(
    string DataFile,
    string LabelColumn,
    string Task,
    string? Name = null,
    int TimeLimitSeconds = 300,
    string Metric = "auto",
    double TestSplit = 0.2,
    int? MaxRows = null)
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
