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

    /// <summary>The temp project root backing this factory's <c>models/</c>, <c>datasets/</c>, etc. —
    /// exposed so tests can seed a real production model (model.zip/metadata.json/config.json)
    /// before exercising an endpoint end-to-end.</summary>
    public string TestProjectRoot => _testProjectRoot;

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

            // Add test-specific ProjectDiscovery pinned to our temp directory. Deliberately NOT via
            // Environment.CurrentDirectory: that is process-global state, and xUnit runs test classes
            // (= factory instances) in parallel — the old CWD mutation let one factory's Dispose
            // delete the directory another factory's Program startup was using as its working
            // directory, so the entry point died before ever building an IHost. Latent while
            // ApiIntegrationTests was the only factory consumer; exposed when ForecastingApiTests
            // added a second one (D21-A).
            services.AddSingleton<IProjectDiscovery>(new FixedRootProjectDiscovery(_testProjectRoot));

            // Override Ops/DataStore services to use test directory
            ReplaceService<IModelComparer>(services, new FileModelComparer(_testProjectRoot));
            ReplaceService<IRetrainingTrigger>(services, new TimeBasedTrigger(_testProjectRoot));
            ReplaceService<IPredictionLogger>(services, new FilePredictionLogger(_testProjectRoot));
            ReplaceService<IFeedbackCollector>(services, new FileFeedbackCollector(_testProjectRoot));
            ReplaceService<IPromotionManager>(services, new FilePromotionManager(_testProjectRoot));
        });

        // Override authentication for tests (no real JWT needed) — unless a subclass keeps the
        // real JwtBearer pipeline to exercise the unauthenticated-challenge behavior (401 body).
        if (UseTestAuthentication)
        {
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
    }

    /// <summary>When true (default) the factory swaps in an always-succeed test auth handler so
    /// endpoint tests don't need real tokens. A subclass returns false to keep the production
    /// JwtBearer pipeline — needed to verify the real unauthenticated 401 challenge response.</summary>
    protected virtual bool UseTestAuthentication => true;

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

    /// <summary>An <see cref="IProjectDiscovery"/> pinned to a known root — no process-global
    /// CWD dependency, so parallel factories (one per test class) stay isolated.</summary>
    private sealed class FixedRootProjectDiscovery : IProjectDiscovery
    {
        private readonly string _root;

        public FixedRootProjectDiscovery(string root) => _root = root;

        public string FindRoot() => _root;

        public string FindRoot(string startingDirectory) => _root;

        public bool IsProjectRoot(string path) => Directory.Exists(Path.Combine(path, ".mloop"));

        public void EnsureProjectRoot() { }

        public string GetMLoopDirectory(string projectRoot) => Path.Combine(projectRoot, ".mloop");
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
