using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using Microsoft.ML;
using MLoop.API.Caching;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using Serilog;

namespace MLoop.API;

/// <summary>
/// Experiment endpoints: listing, detail, project status, comparison, and evaluation on test data.
/// </summary>
public static class ExperimentEndpoints
{
    public static void MapExperimentEndpoints(this WebApplication app)
    {
        // List experiments endpoint (secured)
        app.MapGet("/experiments", async (
            [FromQuery] string? name,
            IExperimentStore experimentStore,
            IModelRegistry registry,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = string.IsNullOrWhiteSpace(name) ? null : name.Trim().ToLowerInvariant();

            try
            {
                logger.LogInformation("Listing experiments (filter: '{ModelName}')", modelName ?? "all");
                // The rows of `mloop list --json --all`: each experiment's metrics, gate verdict and production flag.
                var (rows, production) = await ExperimentListing.ListAsync(experimentStore, registry, modelName, completedOnly: false, ct);

                logger.LogInformation("Retrieved {Count} experiment(s) in {ElapsedMs}ms",
                    rows.Count, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    count = rows.Count,
                    filter = modelName,
                    experiments = rows,
                    production,
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to list experiments after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to list experiments",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("ListExperiments")
        .WithTags("Experiments")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Get experiment detail endpoint (secured)
        app.MapGet("/experiments/{id}", async (
            string id,
            [FromQuery] string? name,
            IExperimentStore experimentStore,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);

            try
            {
                logger.LogInformation("Loading experiment '{ExperimentId}' for model '{ModelName}'", id, modelName);

                if (!experimentStore.ExperimentExists(modelName, id))
                {
                    logger.LogWarning("Experiment '{ExperimentId}' not found for model '{ModelName}'", id, modelName);
                    return Results.NotFound(new { error = $"Experiment '{id}' not found for model '{modelName}'." });
                }

                var experiment = await experimentStore.LoadAsync(modelName, id, ct);

                logger.LogInformation("Loaded experiment '{ExperimentId}' in {ElapsedMs}ms",
                    id, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    modelName = experiment.ModelName,
                    experimentId = experiment.ExperimentId,
                    timestamp = experiment.Timestamp,
                    status = experiment.Status,
                    task = experiment.Task,
                    config = new
                    {
                        dataFile = experiment.Config.DataFile,
                        dataFileHash = experiment.Config.DataFileHash,
                        labelColumn = experiment.Config.LabelColumn,
                        timeLimitSeconds = experiment.Config.TimeLimitSeconds,
                        autoTime = experiment.Config.AutoTime,
                        metric = experiment.Config.Metric,
                        testSplit = experiment.Config.TestSplit,
                        groupColumn = experiment.Config.GroupColumn,
                        hasSchema = experiment.Config.InputSchema != null
                    },
                    result = experiment.Result != null ? new
                    {
                        bestTrainer = experiment.Result.BestTrainer,
                        trainingTimeSeconds = experiment.Result.TrainingTimeSeconds
                    } : null,
                    metrics = experiment.Metrics
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to load experiment '{ExperimentId}' after {ElapsedMs}ms",
                    id, stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to load experiment",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetExperiment")
        .WithTags("Experiments")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status404NotFound)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Project status endpoint (secured)
        app.MapGet("/status", async (
            IModelRegistry registry,
            IExperimentStore experimentStore,
            ProjectRootPath projectRootPath,
            ConfigLoader configLoader,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();

            try
            {
                logger.LogInformation("Retrieving project status");

                var projectRoot = projectRootPath.Value;
                var projectName = Path.GetFileName(projectRoot);

                // Get all experiments and production models
                var allExperiments = (await experimentStore.ListAsync(null, ct)).ToList();
                var productionModels = (await registry.ListAsync(null, ct)).ToList();

                // Aggregate model statuses
                var modelNames = allExperiments
                    .Select(e => e.ModelName ?? ConfigDefaults.DefaultModelName)
                    .Distinct()
                    .OrderBy(n => n)
                    .ToList();

                var productionDict = productionModels.ToDictionary(m => m.ModelName, m => m);

                var modelStatuses = modelNames.Select(modelName =>
                {
                    var modelExps = allExperiments
                        .Where(e => (e.ModelName ?? ConfigDefaults.DefaultModelName) == modelName)
                        .ToList();

                    var completed = modelExps.Count(e =>
                        e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase));
                    var failed = modelExps.Count(e =>
                        e.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase));

                    productionDict.TryGetValue(modelName, out var prodModel);

                    // Ranked by metric direction, among experiments measured alike (one metric, one weighting).
                    var bestPick = MLoop.CLI.Infrastructure.FileSystem.ExperimentRanking.SelectBest(modelExps);

                    return new
                    {
                        modelName,
                        totalExperiments = modelExps.Count,
                        completed,
                        failed,
                        hasProduction = prodModel != null,
                        productionExperimentId = prodModel?.ExperimentId,
                        bestMetric = bestPick?.Best.BestMetric,
                        bestMetricName = bestPick?.Measurement.Metric,
                        bestMetricWeightColumn = bestPick?.Measurement.Weighting.Column,
                        bestExperimentId = bestPick?.Best.ExperimentId,
                        experimentsNotCompared = bestPick?.NotCompared ?? 0,
                        latestExperiment = modelExps
                            .OrderByDescending(e => e.Timestamp)
                            .FirstOrDefault()?.ExperimentId
                    };
                }).ToList();

                // Check data files
                var datasetsDir = Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory);
                var dataFiles = new
                {
                    trainCsv = File.Exists(Path.Combine(datasetsDir, ProjectLayout.TrainFileName)),
                    testCsv = File.Exists(Path.Combine(datasetsDir, ProjectLayout.TestFileName)),
                    predictCsv = File.Exists(Path.Combine(datasetsDir, ProjectLayout.PredictFileName))
                };

                // Load config if available
                object? configSummary = null;
                try
                {
                    var config = await configLoader.LoadUserConfigAsync();
                    configSummary = config.Models.Select(kvp => new
                    {
                        name = kvp.Key,
                        task = kvp.Value.Task,
                        label = kvp.Value.Label,
                        // An unset budget is auto-estimated from the data when training runs (as `mloop status` says), so it has no number here.
                        timeLimitSeconds = kvp.Value.Training?.TimeLimitSeconds,
                        metric = kvp.Value.Training?.Metric ?? ConfigDefaults.DefaultMetric
                    }).ToList();
                }
                catch { /* Config not available */ }

                logger.LogInformation("Project status retrieved in {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    project = projectName,
                    projectRoot,
                    summary = new
                    {
                        totalModels = modelNames.Count,
                        totalExperiments = allExperiments.Count,
                        completedExperiments = allExperiments.Count(e =>
                            e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)),
                        failedExperiments = allExperiments.Count(e =>
                            e.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase)),
                        productionModels = productionModels.Count
                    },
                    models = modelStatuses,
                    dataFiles,
                    config = configSummary
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve project status after {ElapsedMs}ms",
                    stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to retrieve project status",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetStatus")
        .WithTags("Status")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Compare experiments endpoint (secured)
        app.MapGet("/compare", async (
            [FromQuery] string candidate,
            [FromQuery] string baseline,
            [FromQuery] string? name,
            IModelComparer comparer,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);

            try
            {
                logger.LogInformation("Comparing experiments '{Candidate}' vs '{Baseline}' for model '{ModelName}'",
                    candidate, baseline, modelName);

                var result = await comparer.CompareAsync(modelName, candidate, baseline, ct);

                logger.LogInformation("Comparison completed in {ElapsedMs}ms: candidate is {Better}",
                    stopwatch.ElapsedMilliseconds, result.CandidateIsBetter ? "better" : "worse");

                return Results.Ok(new
                {
                    modelName,
                    candidateExpId = result.CandidateExpId,
                    baselineExpId = result.BaselineExpId,
                    candidateIsBetter = result.CandidateIsBetter,
                    // The one metric the verdict and the scores are on — the run's optimized metric, as promotion judges.
                    metric = result.Metric,
                    candidateScore = result.CandidateScore,
                    baselineScore = result.BaselineScore,
                    improvement = result.Improvement,
                    recommendation = result.Recommendation,
                    metricDetails = result.MetricDetails.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new
                        {
                            candidateValue = kvp.Value.CandidateValue,
                            baselineValue = kvp.Value.BaselineValue,
                            difference = kvp.Value.Difference,
                            isBetter = kvp.Value.IsBetter
                        })
                });
            }
            catch (FileNotFoundException ex)
            {
                logger.LogWarning(ex, "Experiment not found for comparison");
                return Results.NotFound(new { error = ex.Message });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Comparison failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Comparison failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("CompareExperiments")
        .WithTags("Experiments")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status404NotFound)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Evaluate model endpoint (secured)
        app.MapPost("/evaluate", async (
            [FromBody] EvaluateRequest request,
            IExperimentStore experimentStore,
            IModelRegistry registry,
            ProjectRootPath projectRootPath,
            EvaluationEngine evaluationEngine,
            IFileSystemManager fileSystem,
            IProjectDiscovery projectDiscovery,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(request.Name);
            var experimentId = request.ExperimentId;

            try
            {
                logger.LogInformation("Evaluating experiment '{ExperimentId}' for model '{ModelName}'",
                    experimentId, modelName);

                // Resolve experiment: use specified or production
                if (string.IsNullOrWhiteSpace(experimentId))
                {
                    var prod = await registry.GetProductionAsync(modelName, ct);
                    if (prod == null)
                        return Results.Ok(new { status = "skipped", reason = "no_production_model", model = modelName });
                    experimentId = prod.ExperimentId;
                }

                // The named experiment's own model — never production's in its place: a report under the
                // experiment's id must be that experiment's measurement.
                var (modelPath, refusal) = ExperimentModels.Find(experimentStore, modelName, experimentId);
                if (modelPath is null)
                    return Results.NotFound(new { error = refusal });

                var experiment = await experimentStore.LoadAsync(modelName, experimentId, ct);

                // Determine test data path (with path traversal protection)
                var testDataPath = request.TestDataPath;
                if (string.IsNullOrWhiteSpace(testDataPath))
                    return Results.BadRequest(new { error = "testDataPath is required." });

                // A relative path is the project's, as for POST /train.
                var resolvedTestPath = ProjectPaths.Resolve(testDataPath, projectRootPath.Value);
                if (!ProjectPaths.IsWithin(resolvedTestPath, projectRootPath.Value))
                    return Results.BadRequest(new { error = "Test data file must be within the project directory." });

                if (!TabularDataFile.Exists(resolvedTestPath))
                    return Results.BadRequest(new { error = $"Test data file not found: {testDataPath}" });

                // What `mloop evaluate` does before it measures: the file read as a table (Parquet, Excel, JSON, TSV),
                // then checked against the experiment's schema — a refused file is the caller's to fix, not a 500.
                var trainingDataWarning = await ModelComparison.TrainingDataWarningAsync(experiment, resolvedTestPath, ct);
                resolvedTestPath = await TabularDataFile.AsCsvAsync(resolvedTestPath);
                var validation = await new SchemaValidator(fileSystem, projectDiscovery).ValidateAsync(resolvedTestPath, modelName, experimentId);
                if (!validation.IsValid)
                    return Results.BadRequest(new
                    {
                        error = "Schema validation failed. " + (validation.ErrorMessage
                            ?? "Missing columns: " + string.Join(", ", validation.MissingColumns)),
                        missingColumns = validation.MissingColumns,
                        suggestions = validation.Suggestions,
                    });
                var warnings = new List<string>(validation.Warnings);
                if (trainingDataWarning is not null)
                    warnings.Insert(0, trainingDataWarning);

                var labelColumn = experiment.Config.LabelColumn;
                var taskType = experiment.Task;

                // The rows count by the same weight `mloop evaluate` would give them — the one asked for, else
                // the column the experiment was trained with when the file has it.
                var (weightColumn, weightWarning) = ModelComparison.ResolveWeight(request.Weight, experiment, resolvedTestPath);
                if (weightWarning is not null)
                    warnings.Add(weightWarning);

                ComparisonBaseline? baseline = null;
                if (!string.IsNullOrWhiteSpace(request.Against))
                {
                    baseline = await ModelComparison.ResolveBaselineAsync(request.Against.Trim(), modelName, experimentId, experiment,
                        experimentStore, registry, ct);
                    if (baseline.Error is { } error)
                        return baseline.NotFound ? Results.NotFound(new { error }) : Results.BadRequest(new { error });
                }

                var (metrics, confusion) = await evaluationEngine.EvaluateAsync(
                    modelPath, resolvedTestPath, labelColumn, taskType, ct,
                    experiment.Config.InputSchema, experiment.Config.GroupColumn,
                    experiment.Config.UserColumn, experiment.Config.ItemColumn, weightColumn,
                    new MLoop.Core.Metrics.MetricEngine(projectRootPath.Value, new MLoop.CLI.Infrastructure.Diagnostics.ConsoleLogger()));

                var comparison = baseline is null
                    ? null
                    : await ModelComparison.CompareAsync(evaluationEngine, experimentId, modelPath, experiment, baseline, resolvedTestPath, weightColumn, ct);

                var sliceBy = string.IsNullOrWhiteSpace(request.SliceBy) ? null : request.SliceBy.Trim();
                var slices = sliceBy is null
                    ? null
                    : await EvaluationSlices.EvaluateAsync(evaluationEngine, sliceBy, resolvedTestPath, modelPath, experiment, weightColumn, ct);

                logger.LogInformation("Evaluation completed in {ElapsedMs}ms with {MetricCount} metrics",
                    stopwatch.ElapsedMilliseconds, metrics.Count);

                return Results.Ok(new
                {
                    modelName,
                    experimentId,
                    task = taskType,
                    testDataPath,
                    weightColumn,
                    metrics,
                    confusion,
                    comparison,
                    sliceBy,
                    slices,
                    warnings = warnings.Count == 0 ? null : warnings,
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Evaluation failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Evaluation failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("EvaluateModel")
        .WithTags("Experiments")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status400BadRequest)
        .Produces<object>(StatusCodes.Status404NotFound)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization("Admin");
    }
}

/// <summary>A <c>POST /evaluate</c> body — the options of <c>mloop evaluate</c>; <c>SliceBy</c> is <c>--slice-by</c>.</summary>
public record EvaluateRequest(
    string TestDataPath,
    string? ExperimentId = null,
    string? Name = null,
    string? Weight = null,
    string? Against = null,
    string? SliceBy = null);
