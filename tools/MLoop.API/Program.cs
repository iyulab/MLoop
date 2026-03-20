using Microsoft.ML;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using AspNetCoreRateLimit;
using Serilog;
using Serilog.Events;
using System.Diagnostics;
using System.Reflection;

// Configure Serilog
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Information()
    .MinimumLevel.Override("Microsoft.AspNetCore", LogEventLevel.Warning)
    .Enrich.FromLogContext()
    .Enrich.WithProperty("Application", "MLoop.API")
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj} {Properties:j}{NewLine}{Exception}")
    .WriteTo.File(
        path: "logs/mloop-api-.log",
        rollingInterval: RollingInterval.Day,
        outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj} {Properties:j}{NewLine}{Exception}",
        retainedFileCountLimit: 30)
    .CreateLogger();

try
{
    Log.Information("Starting MLoop.API application");

var builder = WebApplication.CreateBuilder(args);

// Replace default logging with Serilog
builder.Host.UseSerilog();

// Configure JSON serialization
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals;
});

// Add Swagger services
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new()
    {
        Title = "MLoop MLOps API",
        Version = "v1",
        Description = "Full MLOps REST API for ML.NET models trained with MLoop. " +
            "Supports training, prediction, evaluation, model management, and monitoring. " +
            "All data endpoints require JWT Bearer authentication. " +
            "Use the 'name' query parameter to target specific models (defaults to 'default')."
    });
});

// Add JWT Authentication
var jwtKey = builder.Configuration["Jwt:Key"];
if (string.IsNullOrEmpty(jwtKey))
{
    jwtKey = "MLoop-Default-Secret-Key-Change-In-Production-Min-32-Chars";
    Log.Warning("Using default JWT key. Set Jwt:Key in configuration for production use.");
}
var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? "MLoop.API";
var jwtAudience = builder.Configuration["Jwt:Audience"] ?? "MLoop.Client";

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = jwtIssuer,
            ValidAudience = jwtAudience,
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
        };
    });

builder.Services.AddAuthorization();

// Add Rate Limiting (configurable via appsettings.json "IpRateLimiting" section)
builder.Services.AddMemoryCache();
var rateLimitSection = builder.Configuration.GetSection("IpRateLimiting");
if (rateLimitSection.Exists())
{
    builder.Services.Configure<IpRateLimitOptions>(rateLimitSection);
}
else
{
    builder.Services.Configure<IpRateLimitOptions>(options =>
    {
        options.EnableEndpointRateLimiting = true;
        options.StackBlockedRequests = false;
        options.HttpStatusCode = 429;
        options.RealIpHeader = "X-Real-IP";
        options.GeneralRules =
        [
            new() { Endpoint = "POST:/predict", Period = "1m", Limit = 60 },
            new() { Endpoint = "*", Period = "1m", Limit = 100 }
        ];
    });
}
builder.Services.AddInMemoryRateLimiting();
builder.Services.AddSingleton<IRateLimitConfiguration, RateLimitConfiguration>();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// Register MLoop services
builder.Services.AddSingleton<IFileSystemManager, FileSystemManager>();
builder.Services.AddSingleton<IProjectDiscovery, ProjectDiscovery>();
builder.Services.AddSingleton<ConfigLoader>();
builder.Services.AddSingleton<IModelNameResolver, ModelNameResolver>();
builder.Services.AddSingleton<IExperimentStore, ExperimentStore>();
builder.Services.AddSingleton<IModelRegistry, ModelRegistry>();
builder.Services.AddSingleton<MLContext>();
builder.Services.AddSingleton<EvaluationEngine>();
builder.Services.AddSingleton<TrainingJobStore>();
builder.Services.AddSingleton<MLoop.CLI.Infrastructure.ML.ITrainingEngine>(sp =>
{
    var fileSystem = sp.GetRequiredService<IFileSystemManager>();
    var experimentStore = sp.GetRequiredService<IExperimentStore>();
    var projectRoot = sp.GetRequiredService<IProjectDiscovery>().FindRoot();
    return new TrainingEngine(fileSystem, experimentStore, projectRoot);
});
builder.Services.AddSingleton<TrainingJobRunner>();
builder.Services.AddHostedService<TrainingJobRunner>(sp => sp.GetRequiredService<TrainingJobRunner>());

// Register MLoop.Ops services
builder.Services.AddSingleton<IModelComparer>(sp =>
{
    var projectRoot = sp.GetRequiredService<IProjectDiscovery>().FindRoot()
        ?? Directory.GetCurrentDirectory();
    return new FileModelComparer(projectRoot);
});
builder.Services.AddSingleton<IPromotionManager>(sp =>
{
    var projectRoot = sp.GetRequiredService<IProjectDiscovery>().FindRoot()
        ?? Directory.GetCurrentDirectory();
    var comparer = sp.GetRequiredService<IModelComparer>();
    return new FilePromotionManager(projectRoot, comparer);
});
builder.Services.AddSingleton<IRetrainingTrigger>(sp =>
{
    var projectRoot = sp.GetRequiredService<IProjectDiscovery>().FindRoot()
        ?? Directory.GetCurrentDirectory();
    return new TimeBasedTrigger(projectRoot);
});

// Register MLoop.DataStore services
builder.Services.AddSingleton<IPredictionLogger>(sp =>
{
    var projectRoot = sp.GetRequiredService<IProjectDiscovery>().FindRoot()
        ?? Directory.GetCurrentDirectory();
    return new FilePredictionLogger(projectRoot);
});
builder.Services.AddSingleton<IFeedbackCollector>(sp =>
{
    var projectRoot = sp.GetRequiredService<IProjectDiscovery>().FindRoot()
        ?? Directory.GetCurrentDirectory();
    return new FileFeedbackCollector(projectRoot);
});

var app = builder.Build();

// Add request logging middleware
app.UseSerilogRequestLogging(options =>
{
    options.MessageTemplate = "HTTP {RequestMethod} {RequestPath} responded {StatusCode} in {Elapsed:0.0000}ms";
    options.EnrichDiagnosticContext = (diagnosticContext, httpContext) =>
    {
        diagnosticContext.Set("RequestHost", httpContext.Request.Host.Value ?? "unknown");
        diagnosticContext.Set("RequestScheme", httpContext.Request.Scheme);
        diagnosticContext.Set("UserAgent", httpContext.Request.Headers["User-Agent"].ToString());
        diagnosticContext.Set("ClientIP", httpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown");
    };
});

// Configure Swagger UI
app.UseSwagger();
app.UseSwaggerUI(options =>
{
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "MLoop API v1");
    options.RoutePrefix = string.Empty; // Serve Swagger UI at root
});

app.UseCors();

app.UseIpRateLimiting();

app.UseAuthentication();
app.UseAuthorization();

// Health check endpoint
app.MapGet("/health", (ILogger<Program> logger) =>
{
    logger.LogInformation("Health check requested");
    return Results.Ok(new
    {
        status = "healthy",
        timestamp = DateTime.UtcNow,
        version = typeof(Program).Assembly
            .GetCustomAttribute<System.Reflection.AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion ?? typeof(Program).Assembly.GetName().Version?.ToString() ?? "unknown"
    });
})
.WithName("HealthCheck")
.WithTags("Health")
.Produces<object>(StatusCodes.Status200OK);

// Model info endpoint (secured)
app.MapGet("/info", async (
    [FromQuery] string? name,
    IModelRegistry registry,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim().ToLowerInvariant();

    try
    {
        logger.LogInformation("Retrieving model information for '{ModelName}'", modelName);
        var productionModel = await registry.GetProductionAsync(modelName, ct);

        if (productionModel == null)
        {
            logger.LogWarning("No production model found for '{ModelName}'", modelName);
            return Results.NotFound(new { error = $"No production model found for '{modelName}'. Train and promote a model first." });
        }

        var modelPath = registry.GetProductionPath(modelName);
        var modelFile = Path.Combine(modelPath, "model.zip");
        var metadataFile = Path.Combine(modelPath, "metadata.json");

        object? metadata = null;
        if (File.Exists(metadataFile))
        {
            var json = await File.ReadAllTextAsync(metadataFile, ct);
            metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
        }

        logger.LogInformation("Model information retrieved for '{ModelName}' experiment {ExperimentId} in {ElapsedMs}ms",
            modelName, productionModel.ExperimentId, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            modelName = modelName,
            experimentId = productionModel.ExperimentId,
            promotedAt = productionModel.PromotedAt,
            metrics = productionModel.Metrics,
            task = productionModel.Task,
            bestTrainer = productionModel.BestTrainer,
            metadata
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to retrieve model information for '{ModelName}' after {ElapsedMs}ms",
            modelName, stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Failed to retrieve model information",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("GetModelInfo")
.WithTags("Model")
.Produces<object>(StatusCodes.Status200OK)
.Produces<object>(StatusCodes.Status404NotFound)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// Prediction endpoint (secured)
app.MapPost("/predict", async (
    [FromQuery] string? name,
    JsonElement input,
    IModelRegistry registry,
    IExperimentStore experimentStore,
    MLContext mlContext,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim().ToLowerInvariant();
    var predictionCount = input.ValueKind == JsonValueKind.Array ? input.GetArrayLength() : 1;

    try
    {
        logger.LogInformation("Prediction request for '{ModelName}' with {Count} input(s)", modelName, predictionCount);

        var productionModel = await registry.GetProductionAsync(modelName, ct);

        if (productionModel == null)
        {
            logger.LogWarning("Prediction failed: No production model found for '{ModelName}'", modelName);
            return Results.NotFound(new { error = $"No production model found for '{modelName}'. Train and promote a model first." });
        }

        var modelPath = Path.Combine(registry.GetProductionPath(modelName), "model.zip");

        if (!File.Exists(modelPath))
        {
            logger.LogError("Model file not found at {ModelPath}", modelPath);
            return Results.NotFound(new { error = $"Model file not found: {modelPath}" });
        }

        // Load experiment data to get InputSchema
        var expData = await experimentStore.LoadAsync(modelName, productionModel.ExperimentId, ct);
        var schema = expData?.Config?.InputSchema;

        if (schema == null)
        {
            logger.LogError("InputSchema not available for '{ModelName}' experiment {ExperimentId}. " +
                "This model was trained before schema capture was implemented. Please retrain.",
                modelName, productionModel.ExperimentId);
            return Results.Problem(
                title: "Schema not available",
                detail: $"InputSchema not found for model '{modelName}' experiment {productionModel.ExperimentId}. " +
                    "This model was trained before schema capture was implemented. Please retrain the model.",
                statusCode: StatusCodes.Status500InternalServerError
            );
        }

        // Parse JSON input to dictionary rows
        var rows = ParseJsonInput(input);

        // Run prediction through shared PredictionService
        var taskType = productionModel.Task ?? "regression";
        var labelColumn = expData?.Config?.LabelColumn;
        var predictionService = new PredictionService(mlContext);

        logger.LogDebug("Running prediction for '{ModelName}' with task '{TaskType}', label '{LabelColumn}'",
            modelName, taskType, labelColumn);

        var result = predictionService.Predict(rows, schema, modelPath, taskType, labelColumn);

        logger.LogInformation("Predictions completed for '{ModelName}': {Count} predictions in {ElapsedMs}ms (avg: {AvgMs}ms/prediction)",
            modelName, result.Rows.Count, stopwatch.ElapsedMilliseconds,
            stopwatch.ElapsedMilliseconds / Math.Max(1.0, result.Rows.Count));

        return Results.Ok(new
        {
            modelName,
            experimentId = productionModel.ExperimentId,
            predictedAt = DateTime.UtcNow,
            task = result.TaskType,
            count = result.Rows.Count,
            predictions = result.Rows,
            warnings = result.Warnings
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Prediction failed for '{ModelName}' with {Count} input(s) after {ElapsedMs}ms",
            modelName, predictionCount, stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Prediction failed",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("Predict")
.WithTags("Prediction")
.Produces<object>(StatusCodes.Status200OK)
.Produces<object>(StatusCodes.Status404NotFound)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// List all models endpoint (secured)
app.MapGet("/models", async (
    [FromQuery] string? name,
    IModelRegistry registry,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? null : name.Trim().ToLowerInvariant();

    try
    {
        logger.LogInformation("Listing models (filter: '{ModelName}')", modelName ?? "all");
        var models = await registry.ListAsync(modelName, ct);
        var modelList = models.ToList();

        logger.LogInformation("Retrieved {Count} model(s) in {ElapsedMs}ms", modelList.Count, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            count = modelList.Count,
            filter = modelName,
            models = modelList.Select(m => new
            {
                modelName = m.ModelName,
                experimentId = m.ExperimentId,
                promotedAt = m.PromotedAt,
                task = m.Task,
                bestTrainer = m.BestTrainer,
                metrics = m.Metrics
            })
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to list models after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Failed to list models",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("ListModels")
.WithTags("Model")
.Produces<object>(StatusCodes.Status200OK)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// List experiments endpoint (secured)
app.MapGet("/experiments", async (
    [FromQuery] string? name,
    IExperimentStore experimentStore,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? null : name.Trim().ToLowerInvariant();

    try
    {
        logger.LogInformation("Listing experiments (filter: '{ModelName}')", modelName ?? "all");
        var experiments = await experimentStore.ListAsync(modelName, ct);
        var expList = experiments.ToList();

        logger.LogInformation("Retrieved {Count} experiment(s) in {ElapsedMs}ms",
            expList.Count, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            count = expList.Count,
            filter = modelName,
            experiments = expList.Select(e => new
            {
                modelName = e.ModelName,
                experimentId = e.ExperimentId,
                timestamp = e.Timestamp,
                status = e.Status,
                bestMetric = e.BestMetric,
                metricName = e.MetricName,
                bestTrainer = e.BestTrainer,
                labelColumn = e.LabelColumn,
                trainingTimeSeconds = e.TrainingTimeSeconds
            })
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
    var modelName = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim().ToLowerInvariant();

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
                labelColumn = experiment.Config.LabelColumn,
                timeLimitSeconds = experiment.Config.TimeLimitSeconds,
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
    IProjectDiscovery projectDiscovery,
    ConfigLoader configLoader,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();

    try
    {
        logger.LogInformation("Retrieving project status");

        var projectRoot = projectDiscovery.FindRoot() ?? Directory.GetCurrentDirectory();
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

            var bestMetric = modelExps
                .Where(e => e.BestMetric.HasValue)
                .OrderByDescending(e => e.BestMetric!.Value)
                .FirstOrDefault()?.BestMetric;

            return new
            {
                modelName,
                totalExperiments = modelExps.Count,
                completed,
                failed,
                hasProduction = prodModel != null,
                productionExperimentId = prodModel?.ExperimentId,
                bestMetric,
                latestExperiment = modelExps
                    .OrderByDescending(e => e.Timestamp)
                    .FirstOrDefault()?.ExperimentId
            };
        }).ToList();

        // Check data files
        var datasetsDir = Path.Combine(projectRoot, "datasets");
        var dataFiles = new
        {
            trainCsv = File.Exists(Path.Combine(datasetsDir, "train.csv")),
            testCsv = File.Exists(Path.Combine(datasetsDir, "test.csv")),
            predictCsv = File.Exists(Path.Combine(datasetsDir, "predict.csv"))
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
                timeLimitSeconds = kvp.Value.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds,
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

// Promote experiment to production (secured)
app.MapPost("/promote", async (
    [FromBody] PromoteRequest request,
    IModelRegistry registry,
    IExperimentStore experimentStore,
    IProjectDiscovery projectDiscovery,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(request.Name)
        ? ConfigDefaults.DefaultModelName : request.Name.Trim().ToLowerInvariant();
    var experimentId = request.ExperimentId;

    try
    {
        logger.LogInformation("Promoting experiment '{ExperimentId}' for model '{ModelName}'",
            experimentId, modelName);

        // Verify experiment exists
        if (!experimentStore.ExperimentExists(modelName, experimentId))
        {
            return Results.NotFound(new { error = $"Experiment '{experimentId}' not found for model '{modelName}'." });
        }

        // Load and validate experiment
        var experiment = await experimentStore.LoadAsync(modelName, experimentId, ct);

        if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
        {
            return Results.BadRequest(new
            {
                error = $"Cannot promote experiment with status '{experiment.Status}'. Only completed experiments can be promoted."
            });
        }

        // Check current production for comparison
        var currentProduction = await registry.GetProductionAsync(modelName, ct);
        var previousExpId = currentProduction?.ExperimentId;

        // Backup current production if exists
        var projectRoot = projectDiscovery.FindRoot() ?? Directory.GetCurrentDirectory();
        if (currentProduction != null && request.CreateBackup != false)
        {
            var productionPath = registry.GetProductionPath(modelName);
            if (Directory.Exists(productionPath))
            {
                var backupsDir = Path.Combine(projectRoot, "models", modelName, "backups");
                var backupPath = Path.Combine(backupsDir,
                    $"{currentProduction.ExperimentId}-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}");
                Directory.CreateDirectory(backupsDir);
                CopyDirectoryRecursive(productionPath, backupPath);
                logger.LogInformation("Backed up current production to {BackupPath}", backupPath);
            }
        }

        // Promote
        await registry.PromoteAsync(modelName, experimentId, ct);

        // Record promotion history
        await RecordPromotionHistoryAsync(
            projectRoot, modelName, experimentId, previousExpId, "api-promote");

        logger.LogInformation("Promoted experiment '{ExperimentId}' to production for model '{ModelName}' in {ElapsedMs}ms",
            experimentId, modelName, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            modelName,
            experimentId,
            previousExperimentId = previousExpId,
            promotedAt = DateTime.UtcNow,
            task = experiment.Task,
            metrics = experiment.Metrics,
            bestTrainer = experiment.Result?.BestTrainer
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Promotion failed for '{ModelName}/{ExperimentId}' after {ElapsedMs}ms",
            modelName, experimentId, stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Promotion failed",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("PromoteExperiment")
.WithTags("Model")
.Produces<object>(StatusCodes.Status200OK)
.Produces<object>(StatusCodes.Status400BadRequest)
.Produces<object>(StatusCodes.Status404NotFound)
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
    var modelName = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim().ToLowerInvariant();

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
    IProjectDiscovery projectDiscovery,
    EvaluationEngine evaluationEngine,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(request.Name)
        ? ConfigDefaults.DefaultModelName : request.Name.Trim().ToLowerInvariant();
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
                return Results.NotFound(new { error = $"No production model found for '{modelName}'." });
            experimentId = prod.ExperimentId;
        }

        if (!experimentStore.ExperimentExists(modelName, experimentId))
            return Results.NotFound(new { error = $"Experiment '{experimentId}' not found for model '{modelName}'." });

        var experiment = await experimentStore.LoadAsync(modelName, experimentId, ct);

        // Find model file
        var expPath = experimentStore.GetExperimentPath(modelName, experimentId);
        var modelPath = Path.Combine(expPath, "model.zip");

        if (!File.Exists(modelPath))
        {
            // Check production path
            modelPath = Path.Combine(registry.GetProductionPath(modelName), "model.zip");
            if (!File.Exists(modelPath))
                return Results.NotFound(new { error = $"Model file not found for experiment '{experimentId}'." });
        }

        // Determine test data path (with path traversal protection)
        var testDataPath = request.TestDataPath;
        if (string.IsNullOrWhiteSpace(testDataPath))
            return Results.BadRequest(new { error = "testDataPath is required." });

        var evalProjectRoot = projectDiscovery.FindRoot() ?? Directory.GetCurrentDirectory();
        var resolvedTestPath = Path.GetFullPath(testDataPath);
        if (!resolvedTestPath.StartsWith(Path.GetFullPath(evalProjectRoot), StringComparison.OrdinalIgnoreCase))
            return Results.BadRequest(new { error = "Test data file must be within the project directory." });

        if (!File.Exists(resolvedTestPath))
            return Results.BadRequest(new { error = $"Test data file not found: {testDataPath}" });

        var labelColumn = experiment.Config.LabelColumn;
        var taskType = experiment.Task;

        var metrics = await evaluationEngine.EvaluateAsync(
            modelPath, testDataPath, labelColumn, taskType, ct,
            experiment.Config.InputSchema, experiment.Config.GroupColumn);

        logger.LogInformation("Evaluation completed in {ElapsedMs}ms with {MetricCount} metrics",
            stopwatch.ElapsedMilliseconds, metrics.Count);

        return Results.Ok(new
        {
            modelName,
            experimentId,
            task = taskType,
            testDataPath,
            metrics
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
.RequireAuthorization();

// Prediction logs endpoint (secured)
app.MapGet("/logs", async (
    [FromQuery] string? name,
    [FromQuery] string? from,
    [FromQuery] string? to,
    [FromQuery] int? limit,
    IPredictionLogger predictionLogger,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? null : name.Trim().ToLowerInvariant();

    try
    {
        logger.LogInformation("Retrieving prediction logs (filter: '{ModelName}')", modelName ?? "all");

        if (!TryParseDateRange(from, to, out var fromDate, out var toDate, out var dateError))
            return Results.BadRequest(new { error = dateError });
        var maxEntries = limit ?? 1000;

        var logs = await predictionLogger.GetLogsAsync(modelName, fromDate, toDate, maxEntries, ct);

        logger.LogInformation("Retrieved {Count} log entries in {ElapsedMs}ms",
            logs.Count, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            count = logs.Count,
            filter = modelName,
            logs = logs.Select(l => new
            {
                modelName = l.ModelName,
                experimentId = l.ExperimentId,
                input = l.Input,
                output = l.Output,
                confidence = l.Confidence,
                timestamp = l.Timestamp
            })
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to retrieve logs after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Failed to retrieve prediction logs",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("GetPredictionLogs")
.WithTags("Monitoring")
.Produces<object>(StatusCodes.Status200OK)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// Get feedback endpoint (secured)
app.MapGet("/feedback", async (
    [FromQuery] string? name,
    [FromQuery] string? from,
    [FromQuery] string? to,
    [FromQuery] int? limit,
    IFeedbackCollector feedbackCollector,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim().ToLowerInvariant();

    try
    {
        logger.LogInformation("Retrieving feedback for model '{ModelName}'", modelName);

        if (!TryParseDateRange(from, to, out var fromDate, out var toDate, out var dateError))
            return Results.BadRequest(new { error = dateError });
        var maxEntries = limit ?? 1000;

        var feedback = await feedbackCollector.GetFeedbackAsync(modelName, fromDate, toDate, maxEntries, ct);

        logger.LogInformation("Retrieved {Count} feedback entries in {ElapsedMs}ms",
            feedback.Count, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            count = feedback.Count,
            modelName,
            feedback = feedback.Select(f => new
            {
                predictionId = f.PredictionId,
                modelName = f.ModelName,
                predictedValue = f.PredictedValue,
                actualValue = f.ActualValue,
                source = f.Source,
                timestamp = f.Timestamp
            })
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to retrieve feedback after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Failed to retrieve feedback",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("GetFeedback")
.WithTags("Monitoring")
.Produces<object>(StatusCodes.Status200OK)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// Submit feedback endpoint (secured)
app.MapPost("/feedback", async (
    [FromBody] FeedbackRequest request,
    IFeedbackCollector feedbackCollector,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();

    try
    {
        logger.LogInformation("Recording feedback for prediction '{PredictionId}'", request.PredictionId);

        await feedbackCollector.RecordFeedbackAsync(
            request.PredictionId, request.ActualValue, request.Source, ct);

        logger.LogInformation("Feedback recorded in {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            predictionId = request.PredictionId,
            recorded = true,
            timestamp = DateTime.UtcNow
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to record feedback after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Failed to record feedback",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("SubmitFeedback")
.WithTags("Monitoring")
.Produces<object>(StatusCodes.Status200OK)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// Retraining trigger evaluation endpoint (secured)
app.MapGet("/trigger", async (
    [FromQuery] string? name,
    IRetrainingTrigger trigger,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var modelName = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim().ToLowerInvariant();

    try
    {
        logger.LogInformation("Evaluating retraining triggers for model '{ModelName}'", modelName);

        var conditions = await trigger.GetDefaultConditionsAsync(modelName, ct);
        var evaluation = await trigger.EvaluateAsync(modelName, conditions, ct);

        logger.LogInformation("Trigger evaluation completed in {ElapsedMs}ms: shouldRetrain={ShouldRetrain}",
            stopwatch.ElapsedMilliseconds, evaluation.ShouldRetrain);

        return Results.Ok(new
        {
            modelName,
            shouldRetrain = evaluation.ShouldRetrain,
            recommendedAction = evaluation.RecommendedAction,
            evaluatedAt = evaluation.EvaluatedAt,
            conditions = evaluation.ConditionResults.Select(cr => new
            {
                type = cr.Condition.Type.ToString(),
                name = cr.Condition.Name,
                threshold = cr.Condition.Threshold,
                currentValue = cr.CurrentValue,
                isMet = cr.IsMet,
                details = cr.Details
            })
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Trigger evaluation failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
        return Results.Problem(
            title: "Trigger evaluation failed",
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError
        );
    }
})
.WithName("EvaluateTrigger")
.WithTags("Monitoring")
.Produces<object>(StatusCodes.Status200OK)
.Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
.RequireAuthorization();

// Start training job (async, secured)
app.MapPost("/train", (
    [FromBody] TrainingJobRequest request,
    TrainingJobStore jobStore,
    TrainingJobRunner jobRunner,
    IProjectDiscovery projectDiscovery,
    ILogger<Program> logger) =>
{
    // Validate data file exists and is within project root
    var trainProjectRoot = projectDiscovery.FindRoot() ?? Directory.GetCurrentDirectory();
    var resolvedDataFile = Path.GetFullPath(request.DataFile);
    if (!resolvedDataFile.StartsWith(Path.GetFullPath(trainProjectRoot), StringComparison.OrdinalIgnoreCase))
        return Results.BadRequest(new { error = "Data file must be within the project directory." });
    if (!File.Exists(resolvedDataFile))
        return Results.BadRequest(new { error = $"Data file not found: {request.DataFile}" });

    // Create job and enqueue for background processing
    var jobId = jobStore.CreateJob(request);
    jobRunner.EnqueueJob(jobId);

    logger.LogInformation("Training job '{JobId}' enqueued for model '{ModelName}' with task '{Task}'",
        jobId, request.ModelName, request.Task);

    return Results.Accepted($"/jobs/{jobId}", new
    {
        jobId,
        modelName = request.ModelName,
        status = "Queued",
        message = "Training job created. Poll /jobs/{jobId} for status."
    });
})
.WithName("StartTraining")
.WithTags("Training")
.Produces<object>(StatusCodes.Status202Accepted)
.Produces<object>(StatusCodes.Status400BadRequest)
.RequireAuthorization();

// List all jobs
app.MapGet("/jobs", (
    TrainingJobStore jobStore,
    ILogger<Program> logger) =>
{
    var jobs = jobStore.GetAllJobs();

    return Results.Ok(new
    {
        count = jobs.Count,
        jobs = jobs.Select(j => new
        {
            jobId = j.JobId,
            modelName = j.ModelName,
            task = j.Task,
            status = j.Status.ToString(),
            statusMessage = j.StatusMessage,
            createdAt = j.CreatedAt,
            completedAt = j.CompletedAt,
            experimentId = j.ExperimentId,
            bestTrainer = j.BestTrainer
        })
    });
})
.WithName("ListJobs")
.WithTags("Training")
.Produces<object>(StatusCodes.Status200OK)
.RequireAuthorization();

// Get job status
app.MapGet("/jobs/{id}", (
    string id,
    TrainingJobStore jobStore,
    ILogger<Program> logger) =>
{
    var job = jobStore.GetJob(id);
    if (job == null)
        return Results.NotFound(new { error = $"Job '{id}' not found." });

    return Results.Ok(new
    {
        jobId = job.JobId,
        modelName = job.ModelName,
        dataFile = job.DataFile,
        labelColumn = job.LabelColumn,
        task = job.Task,
        timeLimitSeconds = job.TimeLimitSeconds,
        metric = job.Metric,
        testSplit = job.TestSplit,
        maxRows = job.MaxRows,
        status = job.Status.ToString(),
        statusMessage = job.StatusMessage,
        createdAt = job.CreatedAt,
        completedAt = job.CompletedAt,
        experimentId = job.ExperimentId,
        metrics = job.Metrics,
        bestTrainer = job.BestTrainer
    });
})
.WithName("GetJobStatus")
.WithTags("Training")
.Produces<object>(StatusCodes.Status200OK)
.Produces<object>(StatusCodes.Status404NotFound)
.RequireAuthorization();

    Log.Information("MLoop.API started successfully");
    await app.RunAsync();
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application terminated unexpectedly");
}
finally
{
    Log.Information("Shutting down MLoop.API");
    await Log.CloseAndFlushAsync();
}

static bool TryParseDateRange(string? from, string? to,
    out DateTimeOffset? fromDate, out DateTimeOffset? toDate, out string? error)
{
    fromDate = null;
    toDate = null;
    error = null;

    if (from != null)
    {
        if (!DateTimeOffset.TryParse(from, out var parsedFrom))
        {
            error = $"Invalid 'from' date format: {from}";
            return false;
        }
        fromDate = parsedFrom;
    }

    if (to != null)
    {
        if (!DateTimeOffset.TryParse(to, out var parsedTo))
        {
            error = $"Invalid 'to' date format: {to}";
            return false;
        }
        toDate = parsedTo;
    }

    return true;
}

static void CopyDirectoryRecursive(string source, string destination)
{
    Directory.CreateDirectory(destination);
    foreach (var file in Directory.GetFiles(source))
        File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: true);
    foreach (var dir in Directory.GetDirectories(source))
        CopyDirectoryRecursive(dir, Path.Combine(destination, Path.GetFileName(dir)));
}

static async Task RecordPromotionHistoryAsync(
    string projectRoot, string modelName, string experimentId,
    string? previousExpId, string action)
{
    var historyPath = Path.Combine(projectRoot, "models", modelName, "promotion-history.json");
    Directory.CreateDirectory(Path.GetDirectoryName(historyPath)!);

    var records = new List<Dictionary<string, object?>>();
    if (File.Exists(historyPath))
    {
        var json = await File.ReadAllTextAsync(historyPath);
        records = JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(json) ?? [];
    }

    records.Add(new Dictionary<string, object?>
    {
        ["modelName"] = modelName,
        ["experimentId"] = experimentId,
        ["previousExperimentId"] = previousExpId,
        ["action"] = action,
        ["timestamp"] = DateTimeOffset.UtcNow
    });

    var options = new JsonSerializerOptions { WriteIndented = true };
    await File.WriteAllTextAsync(historyPath, JsonSerializer.Serialize(records, options));
}

static Dictionary<string, object>[] ParseJsonInput(JsonElement input)
{
    var items = new List<JsonElement>();
    if (input.ValueKind == JsonValueKind.Array)
        foreach (var item in input.EnumerateArray()) items.Add(item);
    else
        items.Add(input);

    return items.Select(item =>
    {
        var row = new Dictionary<string, object>();
        foreach (var prop in item.EnumerateObject())
        {
            row[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.Number => prop.Value.GetDouble(),
                JsonValueKind.String => prop.Value.GetString() ?? string.Empty,
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => prop.Value.ToString()
            };
        }
        return row;
    }).ToArray();
}

public record PromoteRequest(
    string ExperimentId,
    string? Name = null,
    bool? CreateBackup = true);

public record EvaluateRequest(
    string TestDataPath,
    string? ExperimentId = null,
    string? Name = null);

public record FeedbackRequest(
    string PredictionId,
    object ActualValue,
    string? Source = null);

namespace MLoop.API
{
    // Class to make Program accessible to WebApplicationFactory in tests
    public class ProgramTests
    {
    }
}
