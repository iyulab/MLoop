using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;

namespace MLoop.API.Tests;

/// <summary>
/// The request limit is per caller — the token's subject — so callers behind one address keep their own allowance;
/// health probes are not counted; 0 turns it off.
/// </summary>
public class CallerRateLimitTests : IClassFixture<CallerRateLimitTests.ThreeAMinuteFactory>
{
    private readonly ThreeAMinuteFactory _factory;

    public CallerRateLimitTests(ThreeAMinuteFactory factory) => _factory = factory;

    private HttpClient ClientFor(string subject)
    {
        var client = _factory.CreateClient();
        client.DefaultRequestHeaders.Add(TestAuthHandler.SubjectHeader, subject);
        return client;
    }

    [Fact]
    public async Task ACallerPastItsLimit_IsRefusedWithRetryAfter_WhileAnotherCallerIsNot()
    {
        var busy = ClientFor("busy-service");
        for (var i = 0; i < 3; i++)
            (await busy.GetAsync("/models")).StatusCode.Should().NotBe(HttpStatusCode.TooManyRequests);

        var refused = await busy.GetAsync("/models");

        refused.StatusCode.Should().Be(HttpStatusCode.TooManyRequests);
        refused.Headers.RetryAfter.Should().NotBeNull();
        (await refused.Content.ReadFromJsonAsync<Dictionary<string, string>>())!["error"]
            .Should().Contain(CallerRateLimit.PermitsKey);
        (await ClientFor("other-service").GetAsync("/models")).StatusCode.Should().NotBe(HttpStatusCode.TooManyRequests);
    }

    [Fact]
    public async Task HealthProbes_AreNotCounted()
    {
        var probe = ClientFor("probe");
        for (var i = 0; i < 6; i++)
            (await probe.GetAsync("/health")).StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Theory]
    [InlineData(null, CallerRateLimit.DefaultPermitsPerMinute)]
    [InlineData("0", 0)]
    [InlineData("1000", 1000)]
    public void PermitsPerMinute_ReadsTheSetting(string? value, int expected)
    {
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(value is null ? [] : [KeyValuePair.Create(CallerRateLimit.PermitsKey, (string?)value)])
            .Build();

        CallerRateLimit.PermitsPerMinute(configuration).Should().Be(expected);
    }

    [Fact]
    public void PermitsPerMinute_RefusesANegativeLimit()
    {
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection([KeyValuePair.Create(CallerRateLimit.PermitsKey, (string?)"-1")])
            .Build();

        var act = () => CallerRateLimit.PermitsPerMinute(configuration);

        act.Should().Throw<InvalidOperationException>();
    }

    public sealed class ThreeAMinuteFactory : TestWebApplicationFactory
    {
        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            base.ConfigureWebHost(builder);
            builder.UseSetting(CallerRateLimit.PermitsKey, "3");
        }
    }
}
