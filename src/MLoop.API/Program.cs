using Microsoft.ML;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using AspNetCoreRateLimit;
using Serilog;
using Serilog.Events;
using System.Diagnostics;

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
        Description = "REST API for serving ML.NET models trained with MLoop"
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

// Add Rate Limiting
builder.Services.AddMemoryCache();
builder.Services.Configure<IpRateLimitOptions>(options =>
{
    options.EnableEndpointRateLimiting = true;
    options.StackBlockedRequests = false;
    options.HttpStatusCode = 429;
    options.RealIpHeader = "X-Real-IP";
    options.GeneralRules = new List<RateLimitRule>
    {
        new RateLimitRule
        {
            Endpoint = "POST:/predict",
            Period = "1m",
            Limit = 60  // 60 requests per minute
        },
        new RateLimitRule
        {
            Endpoint = "*",
            Period = "1m",
            Limit = 100  // 100 requests per minute for all other endpoints
        }
    };
});
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
        version = "0.1.0-alpha"
    });
})
.WithName("HealthCheck")
.WithTags("Health")
.Produces<object>(StatusCodes.Status200OK);

// Model info endpoint (secured)
app.MapGet("/info", async (IModelRegistry registry, ILogger<Program> logger, CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    try
    {
        logger.LogInformation("Retrieving production model information");
        var productionModel = await registry.GetAsync(ModelStage.Production, ct);

        if (productionModel == null)
        {
            logger.LogWarning("No production model found");
            return Results.NotFound(new { error = "No production model found. Train and promote a model first." });
        }

        var modelPath = registry.GetModelPath(ModelStage.Production);
        var modelFile = Path.Combine(modelPath, "model.zip");
        var metadataFile = Path.Combine(modelPath, "metadata.json");

        object? metadata = null;
        if (File.Exists(metadataFile))
        {
            var json = await File.ReadAllTextAsync(metadataFile, ct);
            metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
        }

        logger.LogInformation("Model information retrieved successfully for experiment {ExperimentId} in {ElapsedMs}ms",
            productionModel.ExperimentId, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            experimentId = productionModel.ExperimentId,
            promotedAt = productionModel.PromotedAt,
            metrics = productionModel.Metrics,
            modelPath = modelFile,
            metadata
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to retrieve model information after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
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
    JsonElement input,
    IModelRegistry registry,
    MLContext mlContext,
    ILogger<Program> logger,
    CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    var predictionCount = input.ValueKind == JsonValueKind.Array ? input.GetArrayLength() : 1;

    try
    {
        logger.LogInformation("Prediction request received for {Count} input(s)", predictionCount);

        var productionModel = await registry.GetAsync(ModelStage.Production, ct);

        if (productionModel == null)
        {
            logger.LogWarning("Prediction failed: No production model found");
            return Results.NotFound(new { error = "No production model found. Train and promote a model first." });
        }

        var modelPath = Path.Combine(registry.GetModelPath(ModelStage.Production), "model.zip");

        if (!File.Exists(modelPath))
        {
            logger.LogError("Model file not found at {ModelPath}", modelPath);
            return Results.NotFound(new { error = $"Model file not found: {modelPath}" });
        }

        // Load model
        var loadStart = Stopwatch.StartNew();
        ITransformer model;
        using (var stream = File.OpenRead(modelPath))
        {
            model = mlContext.Model.Load(stream, out var modelSchema);
        }
        logger.LogDebug("Model loaded in {LoadTimeMs}ms", loadStart.ElapsedMilliseconds);

        // Handle both single object and array of objects
        var predictions = new List<Dictionary<string, object>>();

        if (input.ValueKind == JsonValueKind.Array)
        {
            // Batch prediction
            foreach (var item in input.EnumerateArray())
            {
                var prediction = MakePrediction(mlContext, model, item);
                predictions.Add(prediction);
            }
        }
        else
        {
            // Single prediction
            var prediction = MakePrediction(mlContext, model, input);
            predictions.Add(prediction);
        }

        logger.LogInformation("Predictions completed successfully: {Count} predictions in {ElapsedMs}ms (avg: {AvgMs}ms/prediction)",
            predictions.Count, stopwatch.ElapsedMilliseconds,
            stopwatch.ElapsedMilliseconds / (double)predictions.Count);

        return Results.Ok(new
        {
            experimentId = productionModel.ExperimentId,
            predictedAt = DateTime.UtcNow,
            count = predictions.Count,
            predictions = predictions.Count == 1 ? (object)predictions[0] : predictions
        });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Prediction failed for {Count} input(s) after {ElapsedMs}ms",
            predictionCount, stopwatch.ElapsedMilliseconds);
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
app.MapGet("/models", async (IModelRegistry registry, ILogger<Program> logger, CancellationToken ct) =>
{
    var stopwatch = Stopwatch.StartNew();
    try
    {
        logger.LogInformation("Listing all models");
        var models = await registry.ListAsync(ct);
        var modelList = models.ToList();

        logger.LogInformation("Retrieved {Count} models in {ElapsedMs}ms", modelList.Count, stopwatch.ElapsedMilliseconds);

        return Results.Ok(new
        {
            count = modelList.Count,
            models = modelList.Select(m => new
            {
                stage = m.Stage.ToString().ToLowerInvariant(),
                experimentId = m.ExperimentId,
                promotedAt = m.PromotedAt,
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

static Dictionary<string, object> MakePrediction(MLContext mlContext, ITransformer model, JsonElement input)
{
    // Convert JSON to dynamic data view
    var features = new Dictionary<string, object>();

    foreach (var property in input.EnumerateObject())
    {
        features[property.Name] = property.Value.ValueKind switch
        {
            JsonValueKind.Number => property.Value.GetDouble(),
            JsonValueKind.String => property.Value.GetString() ?? string.Empty,
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => property.Value.ToString()
        };
    }

    // Create prediction engine and predict
    // Note: This is a simplified implementation
    // In production, you'd need schema-aware prediction based on model metadata
    var predictionEngine = mlContext.Model.CreatePredictionEngine<DynamicData, DynamicPrediction>(model);
    var inputData = new DynamicData { Features = features };
    var result = predictionEngine.Predict(inputData);

    return new Dictionary<string, object>
    {
        ["score"] = result.Score,
        ["input"] = features
    };
}

// Dynamic classes for flexible prediction
public class DynamicData
{
    public Dictionary<string, object> Features { get; set; } = new();
}

public class DynamicPrediction
{
    public float Score { get; set; }
}

namespace MLoop.API
{
    // Class to make Program accessible to WebApplicationFactory in tests
    public class ProgramTests
    {
    }
}
