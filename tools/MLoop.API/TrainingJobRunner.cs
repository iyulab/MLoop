using MLoop.API;
using MLoop.CLI.Infrastructure.ML;
using MLoop.CLI.Infrastructure.Configuration;
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

        // Jobs a previous server process accepted and never started: the 202 promised to train them.
        foreach (var queued in jobStore.GetQueuedJobs())
            EnqueueJob(queued.JobId);
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

    /// <summary>
    /// Promotes the job's experiment when it beats production on the metric the job optimized —
    /// the same comparison <c>mloop train</c> makes. This used to pass the first key of the result's
    /// metric dictionary, which for binary classification is always <c>accuracy</c>, so a job
    /// trained for AUC was promoted (or not) on accuracy.
    /// </summary>
    public static Task<bool> PromoteIfBetterAsync(
        IModelRegistry registry,
        TrainingConfig config,
        TrainingResult result,
        CancellationToken cancellationToken)
        => registry.AutoPromoteAsync(config.ModelName, result.ExperimentId, config.CanonicalMetric, cancellationToken);

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

            // The project's model definition, as `mloop train` reads it — columns, weight, the
            // ranking/recommendation/time-series columns — with the request in the command line's place.
            var (definition, refusal) = await TrainingDefinition.ResolveAsync(
                scope.ServiceProvider.GetRequiredService<ConfigLoader>(),
                job.ModelName, job.Task, job.LabelColumn, job.TimeLimitSeconds, job.Metric, job.TestSplit, job.Weight, stoppingToken);
            if (definition is null)
                throw new InvalidOperationException(refusal);
            // The data the job trains on, prepared as `mloop train` prepares its own: more files merged after
            // the first (a file without the weight column counts 1), then read as a table, sampled to
            // MaxRows, rows without a label dropped.
            var projectRoot = scope.ServiceProvider.GetRequiredService<ProjectRootPath>().Value;
            var dataFile = job.DataFile;
            if (job.AdditionalDataFiles.Count > 0)
            {
                var merge = await TrainingDataFiles.MergeAsync(
                    [Path.GetFullPath(job.DataFile), .. job.AdditionalDataFiles.Select(Path.GetFullPath)],
                    definition.Weight,
                    Path.Combine(projectRoot, ".mloop", "merged", $"{job.JobId}.csv"));
                dataFile = merge.Path ?? throw new InvalidOperationException($"{merge.Error} {merge.Guidance}");
            }
            var prepared = await TrainingInput.PrepareAsync(dataFile, definition, job.MaxRows, projectRoot);
            var notes = new List<string>();
            if (prepared.DroppedMissingLabels > 0)
                notes.Add($"{prepared.DroppedMissingLabels} rows without a label dropped");

            // The classes, checked and balanced as `mloop train` does: one class refuses, an imbalance is noted,
            // and --balance splits off the test part before balancing the train part.
            var trainFile = prepared.DataFile;
            string? testFile = null;
            var testSplit = definition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit;
            if (MLoop.Core.AutoML.AutoMLRunner.IsClassification(definition.Task) && !string.IsNullOrEmpty(definition.Label))
            {
                MLoop.Core.Diagnostics.ClassDistributionResult? classes = null;
                try
                {
                    classes = await TrainingInput.ClassesAsync(trainFile, definition.Label);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    notes.Add($"class distribution analysis failed: {ex.Message}");
                }
                if (classes is { Error: null, ClassCount: <= 1 })
                    throw new InvalidOperationException($"{TrainingInput.SingleClassRefusal} {TrainingInput.SingleClassGuidance}");
                if (classes is { Error: null, NeedsAttention: true })
                    notes.Add(classes.Summary);

                if (!string.IsNullOrEmpty(job.Balance))
                {
                    var balanced = TrainingInput.Balance(trainFile, definition.Label, job.Balance, testSplit);
                    if (!string.IsNullOrEmpty(balanced.Result.Message))
                        notes.Add($"balance: {balanced.Result.Message}");
                    (trainFile, testFile) = (balanced.TrainFile, balanced.TestFile);
                }
            }
            if (notes.Count > 0)
                _jobStore.UpdateStatus(job.JobId, JobStatus.Running, "Training started — " + string.Join("; ", notes));

            var config = ModelTrainingConfig.From(definition, job.ModelName, trainFile,
                definition.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds,
                testFile is null ? testSplit : 0,
                testFile is null ? null : Path.GetFullPath(testFile));

            var result = await trainingEngine.TrainAsync(config, cancellationToken: stoppingToken);

            _jobStore.SetResult(job.JobId, result.ExperimentId, result.Metrics, result.BestTrainer);

            // Decided before the job reads Completed, so a client that polls to completion and then reads
            // production sees the outcome — and the job says which it was.
            string? keptReason;
            if (!job.Promote)
                keptReason = "the request asked not to promote (\"promote\": false) — POST /promote promotes it";
            else if (result.HookRefusal is { } hookRefusal)
                // A post-train hook that refused the model keeps it from being promoted, as in `mloop train`.
                keptReason = $"a post-train hook refused it ({hookRefusal})";
            else
                keptReason = await PromoteIfBetterAsync(registry, config, result, stoppingToken)
                    ? null
                    : await registry.KeptInStagingReasonAsync(config.ModelName, result.ExperimentId, config.CanonicalMetric, stoppingToken)
                      ?? "it does not beat production";
            _jobStore.SetPromotion(job.JobId, keptReason is null, keptReason);
            _jobStore.UpdateStatus(job.JobId, JobStatus.Completed,
                $"Training completed: {result.BestTrainer} ({result.TrainingTimeSeconds:F1}s)"
                + (keptReason is null ? " — promoted to production" : $" — kept in staging: {keptReason}"));

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
