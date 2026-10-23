using Microsoft.AspNetCore.Http;
using Serilog.Events;

namespace MLoop.API;

/// <summary>
/// The level a served request is logged at. A health probe that succeeds is <see cref="LogEventLevel.Verbose"/>
/// — below every level a deployment logs — because a container's HEALTHCHECK or a platform's probe calls it every
/// few seconds for as long as the server runs, and one line per probe is most of a log nobody reads. A probe that
/// fails, or one that throws, is logged like any other request.
/// </summary>
public static class RequestLogLevel
{
    /// <summary>The path probes call.</summary>
    public const string HealthPath = "/health";

    public static LogEventLevel For(HttpContext context, double elapsedMs, Exception? exception)
    {
        if (exception is not null || context.Response.StatusCode >= 500)
            return LogEventLevel.Error;
        if (context.Request.Path.Equals(HealthPath, StringComparison.OrdinalIgnoreCase) && context.Response.StatusCode < 400)
            return LogEventLevel.Verbose;
        return LogEventLevel.Information;
    }
}
