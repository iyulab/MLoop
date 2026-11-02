using Microsoft.AspNetCore.Http;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// A query or path value the endpoint cannot use is the caller's mistake: a 4xx, or an empty answer where "nothing matches" is a
/// fair reading — never a 500. Every GET endpoint is sent the values a careless caller sends: a name that is a path, a name
/// that is blank, numbers below zero and not numbers at all, and a date that is not one.
/// </summary>
public class QueryValueApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly HttpClient _client;

    public QueryValueApiTests(TestWebApplicationFactory factory) => _client = factory.CreateClient();

    public static TheoryData<string> Urls()
    {
        var data = new TheoryData<string>();
        foreach (var name in new[] { "..", "..%2F..", "a%2Fb", "%20", "", "%00", "x%5Cy", "default%20", "ÿ" })
        {
            foreach (var endpoint in new[] { "/info", "/models", "/experiments", "/logs", "/feedback", "/drift", "/trigger", "/promotions", "/status" })
                data.Add($"{endpoint}?name={name}");
            data.Add($"/compare?candidate={name}&baseline={name}");
            data.Add($"/compare?candidate=exp-001&baseline={name}&name={name}");
            // The server itself refuses a NUL in the path, before the pipeline; the query carries it here.
            data.Add($"/experiments/{(name.Length == 0 || name == "%00" ? "x" : name)}?name={name}");
        }
        foreach (var number in new[] { "-1", "0", "abc", "99999999999999999999", "1.5", "" })
        {
            data.Add($"/logs?limit={number}");
            data.Add($"/drift?rows={number}");
            data.Add($"/feedback?limit={number}");
        }
        foreach (var date in new[] { "garbage", "2026-13-45", "0001-01-01", "9999-12-31", "", "2026-11-02T99:99" })
            data.Add($"/logs?from={date}&to={date}");
        data.Add("/drift?sliceBy=%00");
        data.Add("/drift?sliceBy=nonexistent");
        data.Add("/experiments?name=default&limit=-5");
        data.Add("/jobs/..%2F..");
        data.Add("/jobs/a%0Ab");
        data.Add("/experiments/a%0Ab?name=default");
        data.Add("/jobs/x");
        return data;
    }

    [Theory]
    [MemberData(nameof(Urls))]
    public async Task AValueTheEndpointCannotUse_IsNeverA500(string url)
    {
        var response = await _client.GetAsync(url);

        ((int)response.StatusCode).Should().BeLessThan(500, $"GET {url} answered {(int)response.StatusCode}: {Truncate(await response.Content.ReadAsStringAsync())}");
    }

    private static string Truncate(string text) => text.Length <= 300 ? text : text[..300];
}

/// <summary>The guard that refuses control characters in path and query values, and only there.</summary>
public class ControlCharacterGuardTests
{
    [Theory]
    [InlineData("/experiments", "name=%00", "the query value of 'name'")]
    [InlineData("/logs", "limit=1%0A", "the query value of 'limit'")]
    [InlineData("/info", "na%00me=x", "the query name")]
    public void ControlCharacterInQuery_IsNamed(string path, string query, string expected)
    {
        var context = new DefaultHttpContext();
        context.Request.Path = path;
        context.Request.QueryString = new QueryString("?" + query);

        Assert.StartsWith(expected, ControlCharacterGuard.Offender(context.Request));
    }

    [Fact]
    public void ControlCharacterInPath_IsNamed()
    {
        var context = new DefaultHttpContext();
        context.Request.Path = "/experiments/exp\0001";

        Assert.Equal("the path", ControlCharacterGuard.Offender(context.Request));
    }

    [Theory]
    [InlineData("/experiments", "name=default&limit=5")]
    [InlineData("/logs", "from=2026-11-02T10%3A00&to=")]
    [InlineData("/drift", "sliceBy=%ED%95%9C%EA%B8%80")]
    public void OrdinaryValues_PassIncludingNonAscii(string path, string query)
    {
        var context = new DefaultHttpContext();
        context.Request.Path = path;
        context.Request.QueryString = new QueryString("?" + query);

        Assert.Null(ControlCharacterGuard.Offender(context.Request));
    }
}
