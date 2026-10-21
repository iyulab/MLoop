using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.API;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.API.Tests;

public class RetentionServiceTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;

    public RetentionServiceTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        // Started now, so the server's own first pass has run before a test writes its files.
        _ = factory.Services;
    }

    private RetentionService Service() => new(
        _factory.Services.GetRequiredService<ConfigLoader>(),
        _factory.Services.GetRequiredService<ProjectRootPath>(),
        NullLogger<RetentionService>.Instance);

    private string OldLog()
    {
        var path = Path.Combine(_factory.TestProjectRoot, ".mloop", "logs", "default", "2001-01-01.jsonl");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "{}\n");
        return path;
    }

    private void Yaml(string retention) => File.WriteAllText(Path.Combine(_factory.TestProjectRoot, "mloop.yaml"),
        $"project: t\nmodels:\n  default:\n    task: regression\n    label: y\n{retention}");

    [Fact]
    public async Task WithoutARetentionPeriod_NothingIsRemoved()
    {
        Yaml("");
        var log = OldLog();

        (await Service().PruneOnceAsync()).Should().BeNull();

        File.Exists(log).Should().BeTrue();
    }

    [Fact]
    public async Task TheYamlsPeriod_RemovesOlderDays()
    {
        Yaml("retention:\n  days: 30\n");
        var log = OldLog();

        (await Service().PruneOnceAsync())!.LogFiles.Should().Be(1);

        File.Exists(log).Should().BeFalse();
    }

    [Fact]
    public async Task APeriodBelowOneDay_RemovesNothing_AndDoesNotThrow()
    {
        Yaml("retention:\n  days: 0\n");
        var log = OldLog();

        (await Service().PruneOnceAsync()).Should().BeNull();

        File.Exists(log).Should().BeTrue();
    }
}
