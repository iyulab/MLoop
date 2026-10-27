using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using Microsoft.ML;
using MLoop.API;
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
using Serilog.Events;

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

    // Numbers in responses and logs must not change shape with the host locale.
    MLoop.Core.Globalization.OutputCulture.Pin();

    var builder = WebApplication.CreateBuilder(args);

    // Register the DeepLearning module for object-detection etc. tasks.
    // Production wiring: only apps that reference + register MLoop.Core.DeepLearning
    // get DL support; tabular-only consumers never pull it in.
    MLoop.Core.Training.DeepLearningRegistry.Register(new MLoop.Core.DeepLearning.DeepLearningModule());

    // Replace default logging with Serilog
    builder.Host.UseSerilog();

    // Configure JSON serialization
    builder.Services.ConfigureHttpJsonOptions(options =>
    {
        options.SerializerOptions.NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals;
        // A field without a value is absent, on every endpoint — the same rule every `--json` emitter in
        // the CLI applies. Until 0.32.0 the server wrote such fields as null, so the one response a
        // consumer might read from both surfaces (a prediction row) had two shapes and the response
        // contract had to spend a paragraph telling readers to treat "absent" and "null" as the same
        // thing. One rule per product, not one per surface. See docs/PREDICT-RESPONSE.md.
        options.SerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
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

        const string bearerSchemeId = "bearer";
        c.AddSecurityDefinition(bearerSchemeId, new Microsoft.OpenApi.OpenApiSecurityScheme
        {
            Type = Microsoft.OpenApi.SecuritySchemeType.Http,
            Scheme = "bearer",
            BearerFormat = "JWT",
            Description = "Enter your JWT token. Admin endpoints (promote, evaluate, train) require 'admin' role claim."
        });

        c.AddSecurityRequirement(document => new Microsoft.OpenApi.OpenApiSecurityRequirement
        {
            [new Microsoft.OpenApi.OpenApiSecuritySchemeReference(bearerSchemeId, document)] = []
        });
    });

    // Add JWT Authentication. The default key/issuer/audience live in MLoop.Core.Security.DevJwtDefaults
    // so the validator here and the issuer (`mloop token`) share one authority and cannot drift.
    var jwtKey = builder.Configuration["Jwt:Key"];
    var isDefaultKey = string.IsNullOrEmpty(jwtKey) || jwtKey == MLoop.Core.Security.DevJwtDefaults.Key;

    if (isDefaultKey)
    {
        if (builder.Environment.IsProduction())
        {
            Log.Fatal("JWT key is not configured. Set Jwt:Key in environment variables or appsettings.json.");
            Log.Fatal("Example: Jwt__Key=your-secure-random-key-at-least-32-characters");
            throw new InvalidOperationException(
                "Cannot start in Production with default JWT key. Set Jwt:Key in configuration.");
        }

        jwtKey = MLoop.Core.Security.DevJwtDefaults.Key;
        Log.Warning("Using default JWT key — NOT SAFE for production. Set Jwt:Key in configuration.");
    }

    if (jwtKey!.Length < 32)
    {
        Log.Fatal("JWT key must be at least 32 characters. Current length: {Length}", jwtKey.Length);
        throw new InvalidOperationException(
            "JWT key must be at least 32 characters. Set Jwt:Key in configuration.");
    }

    var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? MLoop.Core.Security.DevJwtDefaults.Issuer;
    var jwtAudience = builder.Configuration["Jwt:Audience"] ?? MLoop.Core.Security.DevJwtDefaults.Audience;

    builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
        .AddJwtBearer(options =>
        {
            // Read claims verbatim (no legacy URI remapping) so tokens issued by `mloop token` —
            // which carry short 'sub'/'role' claims — validate and authorize without drift.
            options.MapInboundClaims = false;
            options.TokenValidationParameters = new TokenValidationParameters
            {
                ValidateIssuer = true,
                ValidateAudience = true,
                ValidateLifetime = true,
                ValidateIssuerSigningKey = true,
                ValidIssuer = jwtIssuer,
                ValidAudience = jwtAudience,
                IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey)),
                NameClaimType = "sub",
                RoleClaimType = "role"
            };

            // Replace the default empty 401 body with an actionable hint. A programmatic caller
            // embedding `mloop serve` otherwise gets a bare 401 with no clue that a token is
            // required nor how to mint one — the token-issuance surface is the CLI (`mloop token`),
            // not the server, so it isn't self-evident from the API alone.
            options.Events = new JwtBearerEvents
            {
                OnChallenge = async context =>
                {
                    context.HandleResponse(); // suppress the default challenge (empty body)
                    context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                    context.Response.ContentType = "application/json";
                    await context.Response.WriteAsJsonAsync(new
                    {
                        error = "Unauthorized",
                        hint = "This endpoint requires a JWT bearer token. Issue one with the CLI — " +
                               "`mloop token` for read endpoints (/predict, /info, /logs …), or " +
                               "`mloop token --role admin` for write endpoints (/promote, /train, /evaluate) — " +
                               "then send it as the `Authorization: Bearer <token>` header. See `mloop token --help`."
                    });
                },
                // A valid token that lacks the role (a read token on a write endpoint): say what the endpoint needs,
                // where the default answer is a 403 with nothing in it.
                OnForbidden = async context =>
                {
                    context.Response.StatusCode = StatusCodes.Status403Forbidden;
                    context.Response.ContentType = "application/json";
                    await context.Response.WriteAsJsonAsync(new
                    {
                        error = "Forbidden",
                        hint = "This endpoint needs the admin role (/promote, /rollback, /train, /evaluate, /samples, /cache/*). " +
                               "Issue a token with `mloop token --role admin` and send it as the `Authorization: Bearer <token>` header."
                    });
                }
            };
        });

    builder.Services.AddAuthorization(options =>
    {
        // Admin policy: for destructive/write operations (promote, train, feedback)
        options.AddPolicy("Admin", policy =>
            policy.RequireRole("admin"));

        // ReadOnly policy: for monitoring/read operations (default for all authenticated users)
        options.AddPolicy("ReadOnly", policy =>
            policy.RequireAuthenticatedUser());
    });

    // Requests per minute per caller (token subject; address when there is no token) — see CallerRateLimit.
    CallerRateLimit.Add(builder.Services);

    // Add CORS (configurable via Cors:Origins in appsettings.json)
    var corsOrigins = builder.Configuration.GetSection("Cors:Origins").Get<string[]>();
    builder.Services.AddCors(options =>
    {
        options.AddDefaultPolicy(policy =>
        {
            if (corsOrigins is { Length: > 0 })
            {
                policy.WithOrigins(corsOrigins)
                      .AllowAnyMethod()
                      .AllowAnyHeader();
            }
            else
            {
                // Default: allow all (development-friendly, non-production)
                policy.AllowAnyOrigin()
                      .AllowAnyMethod()
                      .AllowAnyHeader();
            }
        });
    });

    // Register MLoop services
    builder.Services.AddSingleton<IFileSystemManager, FileSystemManager>();
    builder.Services.AddSingleton<IProjectDiscovery, ProjectDiscovery>();
    // The project root is one fact for the life of the process. Register the answer, not the question,
    // so it is resolved once — and so a deployment that cannot find its project fails at startup.
    builder.Services.AddSingleton(sp =>
        new ProjectRootPath(sp.GetRequiredService<IProjectDiscovery>().FindRoot()));
    builder.Services.AddSingleton<ConfigLoader>();
    builder.Services.AddSingleton<IModelNameResolver, ModelNameResolver>();
    builder.Services.AddSingleton<IExperimentStore, ExperimentStore>();
    builder.Services.AddSingleton<IModelRegistry, ModelRegistry>();
    builder.Services.AddSingleton<MLContext>();
    builder.Services.AddSingleton<EvaluationEngine>();
    builder.Services.Configure<ModelCacheOptions>(builder.Configuration.GetSection(ModelCacheOptions.SectionName));
    builder.Services.AddSingleton<IModelCache, ModelCache>();
    builder.Services.AddHostedService<ModelCachePreloader>();
    builder.Services.AddHostedService<RetentionService>();
    // Pretrained encoders for POST /embed — outside any project (MLOOP_ENCODERS or ~/.mloop/encoders) — in the process's
    // pool, which the models that read text through an encoder open theirs from too: each graph opened once and shared.
    // Registered as an instance, so the container does not dispose it: it lives as long as the process.
    builder.Services.AddSingleton(MLoop.Core.Embedding.EmbedderPool.Shared);
    builder.Services.AddSingleton<TrainingJobStore>();
    builder.Services.AddSingleton<MLoop.CLI.Infrastructure.ML.ITrainingEngine>(sp =>
    {
        var fileSystem = sp.GetRequiredService<IFileSystemManager>();
        var experimentStore = sp.GetRequiredService<IExperimentStore>();
        var projectRoot = sp.GetRequiredService<ProjectRootPath>().Value;
        return new TrainingEngine(fileSystem, experimentStore, projectRoot);
    });
    builder.Services.AddSingleton<TrainingJobRunner>();
    builder.Services.AddHostedService<TrainingJobRunner>(sp => sp.GetRequiredService<TrainingJobRunner>());

    // Register MLoop.Ops services
    builder.Services.AddSingleton<IModelComparer>(sp =>
    {
        var projectRoot = sp.GetRequiredService<ProjectRootPath>().Value;
        return new FileModelComparer(projectRoot);
    });
    builder.Services.AddSingleton<IPromotionManager>(sp =>
    {
        var projectRoot = sp.GetRequiredService<ProjectRootPath>().Value;
        return new FilePromotionManager(projectRoot);
    });
    builder.Services.AddSingleton<IRetrainingTrigger>(sp =>
    {
        // The same trigger and default conditions `mloop trigger check` uses.
        var projectRoot = sp.GetRequiredService<ProjectRootPath>().Value;
        return new CompositeRetrainingTrigger(projectRoot, sp.GetRequiredService<IFeedbackCollector>(), sp.GetRequiredService<IPredictionLogger>());
    });

    // Register MLoop.DataStore services
    builder.Services.AddSingleton<IPredictionLogger>(sp =>
    {
        var projectRoot = sp.GetRequiredService<ProjectRootPath>().Value;
        return new FilePredictionLogger(projectRoot);
    });
    builder.Services.AddSingleton<IFeedbackCollector>(sp =>
    {
        var projectRoot = sp.GetRequiredService<ProjectRootPath>().Value;
        return new FileFeedbackCollector(projectRoot);
    });

    var app = builder.Build();
    if (app.Configuration.GetSection(CallerRateLimit.RetiredSection).Exists())
        Log.Warning("{Retired} is no longer read: the limit is per caller, {Key} (now {Permits}; 0 = no limit).",
            CallerRateLimit.RetiredSection, CallerRateLimit.PermitsKey, CallerRateLimit.PermitsPerMinute(app.Configuration));

    // Resolve the project once, here, at startup. Eight service registrations and three endpoints used
    // to each resolve it themselves, and each of those carried a `?? Directory.GetCurrentDirectory()`
    // that could never run — FindRoot throws rather than returning null. So a deployment that could not
    // find its project did not fall back to anything: it started cleanly and then failed with a 500 on
    // whichever request first touched a service that needed the root. Failing here says the same thing
    // while an operator is still watching the log.
    app.Logger.LogInformation(
        "Project root: {ProjectRoot}",
        app.Services.GetRequiredService<ProjectRootPath>().Value);

    // Add request logging middleware
    app.UseSerilogRequestLogging(options =>
    {
        options.MessageTemplate = "HTTP {RequestMethod} {RequestPath} responded {StatusCode} in {Elapsed:0.0000}ms";
        options.GetLevel = RequestLogLevel.For;
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
        options.RoutePrefix = "swagger"; // The UI lives where the serve banner and the guide say it does
    });

    app.UseCors();

    // After authentication, so the limiter knows the caller; before authorization, so a request without a token
    // is counted (by address) before it is refused.
    app.UseAuthentication();
    app.UseRateLimiter();
    app.UseAuthorization();

    // Health check endpoint. HEAD is mapped alongside GET because probes send it: wget --spider,
    // and most load balancers, issue HEAD, and a GET-only route answers those 405 while the service
    // is perfectly healthy. RFC 9110 makes HEAD identical to GET minus the body, so ASP.NET returning
    // the same result here is exactly right — it drops the body itself.
    app.MapMethods(RequestLogLevel.HealthPath, ["GET", "HEAD"], () =>
    {
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

    // Endpoints, one file per group — each map call carries its route metadata.
    app.MapModelEndpoints();
    app.MapPredict();
    app.MapExperimentEndpoints();
    app.MapMonitoringEndpoints();
    app.MapTrainingEndpoints();

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

namespace MLoop.API
{
    // Class to make Program accessible to WebApplicationFactory in tests
    public class ProgramTests
    {
    }
}
