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
    public async Task AdminEndpoint_WithAReadToken_Returns403_SayingWhatItNeeds()
    {
        var token = MLoop.CLI.Commands.TokenCommand.IssueToken(
            MLoop.Core.Security.DevJwtDefaults.Key, "reader", role: null, TimeSpan.FromHours(1));
        using var request = new HttpRequestMessage(HttpMethod.Post, "/promote") { Content = JsonContent.Create(new { }) };
        request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        var response = await _client.SendAsync(request);

        response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        var json = JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync());
        json.GetProperty("error").GetString().Should().Be("Forbidden");
        json.GetProperty("hint").GetString().Should().Contain("admin").And.Contain("mloop token --role admin");
        json.GetProperty("hint").GetString().Should().StartWith("POST /promote needs the admin role", "the answer names the route the caller reached");
    }

    /// <summary>
    /// Every route that wants the admin role says so in its own 403, so no list of them is kept anywhere to fall behind — the
    /// previous one named six and was already missing the dataset and feedback writes.
    /// </summary>
    [Theory]
    [InlineData("POST", "/datasets")]
    [InlineData("DELETE", "/datasets/ds-20260101-000000-00000000")]
    [InlineData("POST", "/experiments/exp-001/export/onnx")]
    [InlineData("POST", "/train")]
    public async Task AnyAdminRoute_Returns403_NamingItself(string method, string url)
    {
        var token = MLoop.CLI.Commands.TokenCommand.IssueToken(
            MLoop.Core.Security.DevJwtDefaults.Key, "reader", role: null, TimeSpan.FromHours(1));
        using var request = new HttpRequestMessage(new HttpMethod(method), url) { Content = JsonContent.Create(new { }) };
        request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        var response = await _client.SendAsync(request);

        response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        var hint = JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync()).GetProperty("hint").GetString();
        hint.Should().StartWith($"{method} {url.Replace("/ds-20260101-000000-00000000", "/{id}").Replace("/exp-001/", "/{id}/")} needs the admin role");
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
