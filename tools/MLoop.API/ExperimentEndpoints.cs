using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.HttpResults;
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
        app.MapGet("/experiments", async Task<Results<Ok<ExperimentListResponse>, ProblemHttpResult>> (
            [FromQuery] string? name,
            IExperimentStore experimentStore,
            IModelRegistry registry,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Filter(name);

            try
            {
                logger.LogInformation("Listing experiments (filter: '{ModelName}')", modelName ?? "all");
                // The rows of `mloop list --json --all`: each experiment's metrics, gate verdict and production flag.
                var (rows, production) = await ExperimentListing.ListAsync(experimentStore, registry, modelName, completedOnly: false, ct);

                logger.LogInformation("Retrieved {Count} experiment(s) in {ElapsedMs}ms",
                    rows.Count, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new ExperimentListResponse(rows.Count, modelName, rows, production));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to list experiments after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to list experiments",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("ListExperiments")
        .WithTags("Experiments")
        .RequireAuthorization();

        // Get experiment detail endpoint (secured)
        app.MapGet("/experiments/{id}", async Task<Results<Ok<ExperimentDetailResponse>, NotFound<ErrorResponse>, ProblemHttpResult>> (
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
                    return TypedResults.NotFound(new ErrorResponse($"Experiment '{id}' not found for model '{modelName}'."));
                }

                var experiment = await experimentStore.LoadAsync(modelName, id, ct);

                logger.LogInformation("Loaded experiment '{ExperimentId}' in {ElapsedMs}ms",
                    id, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new ExperimentDetailResponse(
                    experiment.ModelName,
                    experiment.ExperimentId,
                    experiment.Timestamp,
                    experiment.Status,
                    experiment.Task,
                    new ExperimentConfigView(
                        experiment.Config.DataFile,
                        experiment.Config.DataFileHash,
                        experiment.Config.LabelColumn,
                        experiment.Config.TimeLimitSeconds,
                        experiment.Config.AutoTime,
                        experiment.Config.Metric,
                        experiment.Config.TestSplit,
                        experiment.Config.GroupColumn,
                        HasSchema: experiment.Config.InputSchema != null),
                    experiment.Result != null
                        ? new ExperimentResultView(experiment.Result.BestTrainer, experiment.Result.TrainingTimeSeconds)
                        : null,
                    experiment.Metrics));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to load experiment '{ExperimentId}' after {ElapsedMs}ms",
                    id, stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to load experiment",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetExperiment")
        .WithTags("Experiments")
        .RequireAuthorization();

        // Project status endpoint (secured)
        app.MapGet("/status", async Task<Results<Ok<StatusResponse>, ProblemHttpResult>> (
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
                    .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
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

                    return new ModelStatus(
                        modelName,
                        modelExps.Count,
                        completed,
                        failed,
                        HasProduction: prodModel != null,
                        prodModel?.ExperimentId,
                        bestPick?.Best.BestMetric,
                        bestPick?.Measurement.Metric,
                        bestPick?.Measurement.Weighting.Column,
                        bestPick?.Best.ExperimentId,
                        bestPick?.NotCompared ?? 0,
                        modelExps
                            .OrderByDescending(e => e.Timestamp)
                            .FirstOrDefault()?.ExperimentId);
                }).ToList();

                // Check data files
                var datasetsDir = Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory);
                var dataFiles = new DataFilesStatus(
                    File.Exists(Path.Combine(datasetsDir, ProjectLayout.TrainFileName)),
                    File.Exists(Path.Combine(datasetsDir, ProjectLayout.TestFileName)),
                    File.Exists(Path.Combine(datasetsDir, ProjectLayout.PredictFileName)));

                // Load config if available
                IReadOnlyList<ConfiguredModel>? configSummary = null;
                try
                {
                    var config = await configLoader.LoadUserConfigAsync();
                    configSummary = config.Models.Select(kvp => new ConfiguredModel(
                        kvp.Key,
                        kvp.Value.Task,
                        kvp.Value.Label,
                        // An unset budget is auto-estimated from the data when training runs (as `mloop status` says), so it has no number here.
                        kvp.Value.Training?.TimeLimitSeconds,
                        kvp.Value.Training?.Metric ?? ConfigDefaults.DefaultMetric)).ToList();
                }
                catch { /* Config not available */ }

                logger.LogInformation("Project status retrieved in {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new StatusResponse(
                    projectName,
                    projectRoot,
                    new StatusSummary(
                        modelNames.Count,
                        allExperiments.Count,
                        allExperiments.Count(e =>
                            e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)),
                        allExperiments.Count(e =>
                            e.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase)),
                        productionModels.Count),
                    modelStatuses,
                    dataFiles,
                    configSummary));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve project status after {ElapsedMs}ms",
                    stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to retrieve project status",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetStatus")
        .WithTags("Status")
        .RequireAuthorization();

        // Compare experiments endpoint (secured)
        app.MapGet("/compare", async Task<Results<Ok<CompareResponse>, NotFound<ErrorResponse>, ProblemHttpResult>> (
            [FromQuery] string candidate,
            [FromQuery] string baseline,
            [FromQuery] string? name,
            IModelComparer comparer,
            IExperimentStore experimentStore,
            IModelRegistry registry,
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

                // A better metric is not a promotion: the candidate may be one `POST /promote` refuses (the quality gate,
                // the project's own gate policy). `mloop compare` says so (PromotionAdvice); so does this.
                string? promotionRefusal = null;
                if (result.CandidateIsBetter)
                {
                    var candidateData = await experimentStore.LoadAsync(modelName, candidate, ct);
                    promotionRefusal = ModelRegistry.QualityGateRefusal(candidateData, candidateData.Config?.Metric)
                        ?? await registry.GatePolicyRefusalAsync(modelName, candidate, ct);
                }

                logger.LogInformation("Comparison completed in {ElapsedMs}ms: candidate is {Better}",
                    stopwatch.ElapsedMilliseconds, result.CandidateIsBetter ? "better" : "worse");

                return TypedResults.Ok(new CompareResponse(
                    modelName,
                    result.CandidateExpId,
                    result.BaselineExpId,
                    result.CandidateIsBetter,
                    // The one metric the verdict and the scores are on — the run's optimized metric, as promotion judges.
                    result.Metric,
                    result.CandidateScore,
                    result.BaselineScore,
                    result.Improvement,
                    promotionRefusal is null
                        ? result.Recommendation
                        : $"Do not promote {result.CandidateExpId} - it is better on {result.Metric} but would not be promoted: {promotionRefusal}",
                    promotionRefusal,
                    result.MetricDetails.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new MetricDetail(kvp.Value.CandidateValue, kvp.Value.BaselineValue, kvp.Value.Difference, kvp.Value.IsBetter))));
            }
            catch (FileNotFoundException ex)
            {
                logger.LogWarning(ex, "Experiment not found for comparison");
                return TypedResults.NotFound(new ErrorResponse(ex.Message));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Comparison failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Comparison failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("CompareExperiments")
        .WithTags("Experiments")
        .RequireAuthorization();

        // Evaluate model endpoint (secured)
        app.MapPost("/evaluate", async Task<Results<Ok<EvaluateResponse>, Ok<EvaluateSkippedResponse>, NotFound<ErrorResponse>, BadRequest<ErrorResponse>, BadRequest<SchemaRefusalResponse>, ProblemHttpResult>> (
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
                        return TypedResults.Ok(new EvaluateSkippedResponse("skipped", "no_production_model", modelName));
                    experimentId = prod.ExperimentId;
                }

                // The named experiment's own model — never production's in its place: a report under the
                // experiment's id must be that experiment's measurement.
                var (modelPath, refusal) = ExperimentModels.Find(experimentStore, modelName, experimentId);
                if (modelPath is null)
                    return TypedResults.NotFound(new ErrorResponse(refusal));

                var experiment = await experimentStore.LoadAsync(modelName, experimentId, ct);

                // Determine test data path (with path traversal protection)
                var testDataPath = request.TestDataPath;
                if (string.IsNullOrWhiteSpace(testDataPath))
                    return TypedResults.BadRequest(new ErrorResponse("testDataPath is required."));

                // A relative path is the project's, as for POST /train.
                var resolvedTestPath = ProjectPaths.Resolve(testDataPath, projectRootPath.Value);
                if (!ProjectPaths.IsWithinReal(resolvedTestPath, projectRootPath.Value))
                    return TypedResults.BadRequest(new ErrorResponse("Test data file must be within the project directory."));

                if (!TabularDataFile.Exists(resolvedTestPath))
                    return TypedResults.BadRequest(new ErrorResponse($"Test data file not found: {testDataPath}"));

                // What `mloop evaluate` does before it measures: the file read as a table (Parquet, Excel, JSON, TSV),
                // then checked against the experiment's schema — a refused file is the caller's to fix, not a 500.
                var trainingDataWarning = await ModelComparison.TrainingDataWarningAsync(experiment, resolvedTestPath, ct);
                resolvedTestPath = await TabularDataFile.AsCsvAsync(resolvedTestPath);
                var validation = await new SchemaValidator(fileSystem, projectDiscovery).ValidateAsync(resolvedTestPath, modelName, experimentId);
                if (!validation.IsValid)
                    return TypedResults.BadRequest(new SchemaRefusalResponse(
                        "Schema validation failed. " + (validation.ErrorMessage
                            ?? "Missing columns: " + string.Join(", ", validation.MissingColumns)),
                        validation.MissingColumns,
                        validation.Suggestions));
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
                    {
                        if (baseline.NotFound) return TypedResults.NotFound(new ErrorResponse(error));
                        return TypedResults.BadRequest(new ErrorResponse(error));
                    }
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

                return TypedResults.Ok(new EvaluateResponse(
                    modelName,
                    experimentId,
                    taskType,
                    testDataPath,
                    weightColumn,
                    metrics,
                    confusion,
                    comparison,
                    sliceBy,
                    slices,
                    warnings.Count == 0 ? null : warnings));
            }
            catch (Exception ex) when (ex is MLoop.Core.Training.UnreadableLabelException or MLoop.Core.Data.ColumnLayoutException or MLoop.Core.Data.EmptyTableFileException or MLoop.Core.Data.UnterminatedQuoteException)
            {
                // The file's own fault, as the schema check above: a label written as numbers for people, a file that does not split into
                // columns. `mloop evaluate` says the same and exits 1; a 500 told the caller the server was broken.
                logger.LogInformation("Evaluation refused the test file after {ElapsedMs}ms: {Reason}", stopwatch.ElapsedMilliseconds, ex.Message);
                return TypedResults.BadRequest(new ErrorResponse(ex.Message));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Evaluation failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Evaluation failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("EvaluateModel")
        .WithTags("Experiments")
        .RequireAuthorization("Admin");
    }
}

/// <summary>A <c>POST /evaluate</c> body — the options of <c>mloop evaluate</c>; <c>SliceBy</c> is <c>--slice-by</c>.</summary>
[System.Text.Json.Serialization.JsonUnmappedMemberHandling(System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow)]
public record EvaluateRequest(
    string TestDataPath,
    string? ExperimentId = null,
    string? Name = null,
    string? Weight = null,
    string? Against = null,
    string? SliceBy = null);
