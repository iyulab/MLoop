using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using System.Threading.Channels;

/// <summary>
/// Background service that processes training jobs from a queue.
/// Integrates with ASP.NET Core's hosted service lifecycle for graceful shutdown.
/// </summary>
public class TrainingJobRunner : BackgroundService
{
    private readonly Channel<string> _jobChannel;
    private readonly TrainingJobStore _jobStore;
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<TrainingJobRunner> _logger;

    public TrainingJobRunner(
        TrainingJobStore jobStore,
        IServiceProvider serviceProvider,
        ILogger<TrainingJobRunner> logger)
    {
        _jobChannel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
        {
            SingleReader = true
        });
        _jobStore = jobStore;
        _serviceProvider = serviceProvider;
        _logger = logger;
    }

    /// <summary>
    /// Enqueue a job ID for background processing.
    /// </summary>
    public void EnqueueJob(string jobId)
    {
        _jobChannel.Writer.TryWrite(jobId);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Training job runner started");

        await foreach (var jobId in _jobChannel.Reader.ReadAllAsync(stoppingToken))
        {
            var job = _jobStore.GetJob(jobId);
            if (job == null)
            {
                _logger.LogWarning("Job '{JobId}' not found in store, skipping", jobId);
                continue;
            }

            await ProcessJobAsync(job, stoppingToken);
        }

        _logger.LogInformation("Training job runner stopped");
    }

    private async Task ProcessJobAsync(TrainingJob job, CancellationToken stoppingToken)
    {
        try
        {
            _jobStore.UpdateStatus(job.JobId, JobStatus.Running, "Training started");
            _logger.LogInformation("Training job '{JobId}' started for model '{ModelName}'",
                job.JobId, job.ModelName);

            // Create a scoped service provider for this job
            using var scope = _serviceProvider.CreateScope();
            var trainingEngine = scope.ServiceProvider
                .GetRequiredService<MLoop.CLI.Infrastructure.ML.ITrainingEngine>();
            var registry = scope.ServiceProvider.GetRequiredService<IModelRegistry>();

            var config = new TrainingConfig
            {
                ModelName = job.ModelName,
                DataFile = job.DataFile,
                LabelColumn = job.LabelColumn,
                Task = job.Task,
                TimeLimitSeconds = job.TimeLimitSeconds,
                Metric = job.Metric,
                TestSplit = job.TestSplit
            };

            var result = await trainingEngine.TrainAsync(config, cancellationToken: stoppingToken);

            _jobStore.SetResult(job.JobId, result.ExperimentId, result.Metrics, result.BestTrainer);
            _jobStore.UpdateStatus(job.JobId, JobStatus.Completed,
                $"Training completed: {result.BestTrainer} ({result.TrainingTimeSeconds:F1}s)");

            // Auto-promote if applicable
            var primaryMetric = result.Metrics.Keys.FirstOrDefault() ?? "auto";
            await registry.AutoPromoteAsync(job.ModelName, result.ExperimentId, primaryMetric, stoppingToken);

            _logger.LogInformation("Training job '{JobId}' completed: {BestTrainer}, exp={ExperimentId}",
                job.JobId, result.BestTrainer, result.ExperimentId);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            _jobStore.UpdateStatus(job.JobId, JobStatus.Failed, "Training cancelled: server shutting down");
            _logger.LogWarning("Training job '{JobId}' cancelled due to server shutdown", job.JobId);
        }
        catch (Exception ex)
        {
            _jobStore.UpdateStatus(job.JobId, JobStatus.Failed, ex.Message);
            _logger.LogError(ex, "Training job '{JobId}' failed", job.JobId);
        }
    }
}
