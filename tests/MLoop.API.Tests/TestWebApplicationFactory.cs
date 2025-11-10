using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.API;

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
        });
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
