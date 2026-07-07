using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// Verifies the real JwtBearer challenge: a programmatic caller hitting a protected endpoint with
/// no token must get 401 with an <b>actionable</b> body pointing at <c>mloop token</c>, not the
/// default empty challenge. Uses a factory that keeps the production auth pipeline (the base
/// factory swaps in an always-succeed test handler, which would mask this behavior).
/// </summary>
public class UnauthorizedChallengeTests : IClassFixture<UnauthorizedChallengeTests.RealJwtWebApplicationFactory>
{
    private readonly HttpClient _client;

    public UnauthorizedChallengeTests(RealJwtWebApplicationFactory factory)
        => _client = factory.CreateClient();

    [Fact]
    public async Task Predict_WithoutToken_Returns401_WithActionableHint()
    {
        var response = await _client.PostAsJsonAsync("/predict?name=default", new { });

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);

        var body = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(body);

        json.GetProperty("error").GetString().Should().Be("Unauthorized");
        json.GetProperty("hint").GetString().Should()
            .Contain("mloop token", "the 401 body must tell a programmatic caller how to mint a token")
            .And.Contain("Authorization: Bearer");
    }

    [Fact]
    public async Task Health_IsOpen_WithoutToken()
    {
        var response = await _client.GetAsync("/health");
        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    /// <summary>Keeps the production JwtBearer pipeline (real 401 challenge) while still using the
    /// base factory's temp-project and service overrides so the host can start.</summary>
    public sealed class RealJwtWebApplicationFactory : TestWebApplicationFactory
    {
        protected override bool UseTestAuthentication => false;
    }
}
