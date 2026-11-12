using MLoop.API;

/// <summary>
/// Removes uploaded datasets (<c>POST /datasets</c>) older than <c>Serve:Datasets:RetentionDays</c> while the server runs: at
/// start and then once a day. Off unless the setting is a positive number of days — an upload is training data someone may
/// still mean to use, so nothing is removed by default; one a queued or running job reads is never removed.
/// </summary>
public sealed class DatasetRetentionService : BackgroundService
{
    /// <summary>The setting that turns the sweep on.</summary>
    public const string DaysKey = "Serve:Datasets:RetentionDays";

    private readonly ProjectRootPath _projectRoot;
    private readonly IConfiguration _configuration;
    private readonly TrainingJobStore _jobs;
    private readonly TimeProvider _time;
    private readonly ILogger<DatasetRetentionService> _logger;

    public DatasetRetentionService(
        ProjectRootPath projectRoot, IConfiguration configuration, TrainingJobStore jobs, ILogger<DatasetRetentionService> logger, TimeProvider? time = null)
    {
        _projectRoot = projectRoot;
        _configuration = configuration;
        _jobs = jobs;
        _logger = logger;
        _time = time ?? TimeProvider.System;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(RetentionService.Interval, _time);
        do
        {
            PruneOnce();
        }
        while (await timer.WaitForNextTickAsync(stoppingToken));
    }

    /// <summary>One pass; <c>null</c> when the setting is absent or not a positive number.</summary>
    public DatasetPruneReport? PruneOnce()
    {
        try
        {
            var days = _configuration.GetValue<int?>(DaysKey);
            if (days is null)
                return null;
            if (days < 1)
            {
                _logger.LogWarning("{Key} is {Days}; at least 1 is needed — no upload removed", DaysKey, days);
                return null;
            }

            var report = new DatasetStore(_projectRoot.Value).Prune(days.Value, _time.GetUtcNow(), _jobs);
            if (report.Removed > 0 || report.KeptForJobs > 0)
                _logger.LogInformation("Dataset retention ({Days} days): removed {Removed} upload(s) made before {Cutoff} ({Bytes} bytes); {Kept} kept for a job that reads them",
                    days, report.Removed, report.Cutoff, report.Bytes, report.KeptForJobs);
            return report;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Pruning never takes the server down; the next pass tries again.
            _logger.LogWarning(ex, "Dataset retention pass failed");
            return null;
        }
    }
}
