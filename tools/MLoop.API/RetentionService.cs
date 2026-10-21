using MLoop.API;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.DataStore.Services;

/// <summary>
/// Applies <c>retention.days</c> from mloop.yaml while the server runs: at start and then once a day,
/// removes the prediction logs and feedback older than the period, as <c>mloop logs prune</c> does.
/// mloop.yaml is read on each pass, so a changed period applies without a restart; without one nothing
/// is removed.
/// </summary>
public sealed class RetentionService : BackgroundService
{
    /// <summary>How often the server prunes.</summary>
    public static readonly TimeSpan Interval = TimeSpan.FromDays(1);

    private readonly ConfigLoader _configLoader;
    private readonly ProjectRootPath _projectRoot;
    private readonly TimeProvider _time;
    private readonly ILogger<RetentionService> _logger;

    public RetentionService(ConfigLoader configLoader, ProjectRootPath projectRoot, ILogger<RetentionService> logger, TimeProvider? time = null)
    {
        _configLoader = configLoader;
        _projectRoot = projectRoot;
        _logger = logger;
        _time = time ?? TimeProvider.System;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(Interval, _time);
        do
        {
            await PruneOnceAsync(stoppingToken);
        }
        while (await timer.WaitForNextTickAsync(stoppingToken));
    }

    /// <summary>One pass; <c>null</c> when mloop.yaml sets no usable period.</summary>
    public async Task<RetentionReport?> PruneOnceAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var days = (await _configLoader.LoadUserConfigAsync(cancellationToken)).Retention?.Days;
            if (days is null)
                return null;
            if (days < 1)
            {
                _logger.LogWarning("retention.days is {Days} in mloop.yaml; at least 1 is needed — nothing pruned", days);
                return null;
            }

            var report = RetentionPruner.Prune(_projectRoot.Value, days.Value, _time.GetUtcNow());
            if (report.Files > 0)
                _logger.LogInformation("Retention ({Days} days): removed {LogFiles} log and {FeedbackFiles} feedback day-files before {Cutoff} ({Bytes} bytes)",
                    days, report.LogFiles, report.FeedbackFiles, report.Cutoff, report.Bytes);
            return report;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Pruning never takes the server down; the next pass tries again.
            _logger.LogWarning(ex, "Retention pass failed");
            return null;
        }
    }
}
