using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using MLoop.API;
using Microsoft.AspNetCore.Authentication;
using System.Security.Claims;
using System.Text.Encodings.Web;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace MLoop.API.Tests;

public class TestWebApplicationFactory : WebApplicationFactory<ProgramTests>
{
    private readonly string _testProjectRoot;

    public TestWebApplicationFactory()
    {
        // Create a temporary test project directory with .mloop marker
        _testProjectRoot = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testProjectRoot);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "models", "staging"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "models", "production"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "predictions"));
    }

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureServices(services =>
        {
            // Remove existing ProjectDiscovery and related services
            var descriptor = services.SingleOrDefault(d => d.ServiceType == typeof(IProjectDiscovery));
            if (descriptor != null)
            {
                services.Remove(descriptor);
            }

            // Add test-specific ProjectDiscovery that uses our temp directory
            services.AddSingleton<IProjectDiscovery>(sp =>
            {
                var fileSystem = sp.GetRequiredService<IFileSystemManager>();
                var discovery = new ProjectDiscovery(fileSystem);

                // Override the current directory to our test project root
                Environment.CurrentDirectory = _testProjectRoot;

                return discovery;
            });

            // Override Ops/DataStore services to use test directory
            ReplaceService<IModelComparer>(services, new FileModelComparer(_testProjectRoot));
            ReplaceService<IRetrainingTrigger>(services, new TimeBasedTrigger(_testProjectRoot));
            ReplaceService<IPredictionLogger>(services, new FilePredictionLogger(_testProjectRoot));
            ReplaceService<IFeedbackCollector>(services, new FileFeedbackCollector(_testProjectRoot));
            ReplaceService<IPromotionManager>(services, sp =>
                new FilePromotionManager(_testProjectRoot, sp.GetRequiredService<IModelComparer>()));
        });

        // Override authentication for tests (no real JWT needed)
        builder.ConfigureServices(services =>
        {
            services.AddAuthentication("Test")
                .AddScheme<Microsoft.AspNetCore.Authentication.AuthenticationSchemeOptions, TestAuthHandler>("Test", options => { });
            services.AddAuthorization(options =>
            {
                options.AddPolicy("Admin", policy => policy.RequireRole("admin"));
                options.AddPolicy("ReadOnly", policy => policy.RequireAuthenticatedUser());
            });
        });
    }

    private static void ReplaceService<T>(IServiceCollection services, T implementation) where T : class
    {
        var descriptor = services.SingleOrDefault(d => d.ServiceType == typeof(T));
        if (descriptor != null) services.Remove(descriptor);
        services.AddSingleton<T>(implementation);
    }

    private static void ReplaceService<T>(IServiceCollection services, Func<IServiceProvider, T> factory) where T : class
    {
        var descriptor = services.SingleOrDefault(d => d.ServiceType == typeof(T));
        if (descriptor != null) services.Remove(descriptor);
        services.AddSingleton<T>(factory);
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        if (disposing && Directory.Exists(_testProjectRoot))
        {
            try
            {
                Directory.Delete(_testProjectRoot, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }
}

/// <summary>
/// Test authentication handler that always succeeds without requiring JWT tokens
/// </summary>
public class TestAuthHandler : AuthenticationHandler<AuthenticationSchemeOptions>
{
    public TestAuthHandler(
        IOptionsMonitor<AuthenticationSchemeOptions> options,
        ILoggerFactory logger,
        UrlEncoder encoder)
        : base(options, logger, encoder)
    {
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        var claims = new[]
        {
            new Claim(ClaimTypes.Name, "Test User"),
            new Claim(ClaimTypes.Role, "admin")  // Grant admin role for testing
        };
        var identity = new ClaimsIdentity(claims, "Test");
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, "Test");

        return Task.FromResult(AuthenticateResult.Success(ticket));
    }
}
