using Microsoft.AspNetCore.Http;
using MLoop.API;
using Serilog.Events;

namespace MLoop.API.Tests;

/// <summary>
/// A health probe runs every few seconds for the life of a deployment; logged like a request, it is most of the
/// log. A successful probe is logged below every configured level; anything else keeps its level.
/// </summary>
public class RequestLogLevelTests
{
    private static HttpContext Request(string path, int status)
    {
        var context = new DefaultHttpContext();
        context.Request.Path = path;
        context.Response.StatusCode = status;
        return context;
    }

    [Theory]
    [InlineData("/health", 200, LogEventLevel.Verbose)]
    [InlineData("/HEALTH", 200, LogEventLevel.Verbose)]
    [InlineData("/health", 503, LogEventLevel.Error)]
    [InlineData("/predict", 200, LogEventLevel.Information)]
    [InlineData("/predict", 500, LogEventLevel.Error)]
    [InlineData("/healthz", 200, LogEventLevel.Information)]
    public void A_request_is_logged_at_its_level(string path, int status, LogEventLevel level)
        => Assert.Equal(level, RequestLogLevel.For(Request(path, status), 1.0, null));

    [Fact]
    public void A_probe_that_throws_is_an_error()
        => Assert.Equal(LogEventLevel.Error, RequestLogLevel.For(Request("/health", 200), 1.0, new InvalidOperationException()));
}
