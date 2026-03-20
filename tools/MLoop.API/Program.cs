using Microsoft.ML;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure;
using MLoop.Core.Prediction;
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

// Add Swagger services
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new()
    {
        Title = "MLoop Model Serving API",
        Version = "v1",
        Description = "REST API for serving ML.NET models trained with MLoop. Supports multiple models with the 'name' query parameter."
    });
});

// Add JWT Authentication
var jwtKey = builder.Configuration["Jwt:Key"] ?? "MLoop-Default-Secret-Key-Change-In-Production-Min-32-Chars";
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
            modelPath = modelFile,
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
                metrics = m.Metrics,
                modelPath = m.ModelPath
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

namespace MLoop.API
{
    // Class to make Program accessible to WebApplicationFactory in tests
    public class ProgramTests
    {
    }
}
