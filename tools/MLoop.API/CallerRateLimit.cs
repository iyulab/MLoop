using System.Globalization;
using System.Threading.RateLimiting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace MLoop.API;

/// <summary>
/// How many requests a caller may make per minute — <c>Serve:RateLimit:PermitsPerMinute</c>, default
/// <see cref="DefaultPermitsPerMinute"/>, <c>0</c> for no limit. A caller is the subject of its bearer token: two
/// services behind one address keep their own allowance, and one service keeps its own wherever it calls from. A
/// request without a token (refused by authentication anyway) is counted by its address. Health probes are not
/// counted. A refusal is 429 with <c>Retry-After</c>.
/// </summary>
/// <remarks>
/// Before 0.46 the limit was per address (<c>IpRateLimiting</c>, 60 predictions a minute), so every caller behind
/// one backend shared it and a service predicting faster than one image a second lost predictions to 429. That
/// section is no longer read; the server says so at startup.
/// </remarks>
public static class CallerRateLimit
{
    /// <summary>The configuration key of the limit.</summary>
    public const string PermitsKey = "Serve:RateLimit:PermitsPerMinute";

    /// <summary>The default: ten requests a second, sustained over a minute.</summary>
    public const int DefaultPermitsPerMinute = 600;

    /// <summary>The section the per-address limiter read before 0.46.</summary>
    public const string RetiredSection = "IpRateLimiting";

    /// <summary>The partition a request counts against: its token's subject, else its address.</summary>
    public static string CallerOf(HttpContext context) =>
        context.User.Identity?.IsAuthenticated == true && context.User.FindFirst("sub")?.Value is { Length: > 0 } subject
            ? "sub:" + subject
            : "ip:" + (context.Connection.RemoteIpAddress?.ToString() ?? "unknown");

    /// <summary>The configured limit (0 = none); refuses a negative one.</summary>
    public static int PermitsPerMinute(IConfiguration configuration)
    {
        var permits = configuration.GetValue(PermitsKey, DefaultPermitsPerMinute);
        return permits >= 0
            ? permits
            : throw new InvalidOperationException($"{PermitsKey} must be 0 (no limit) or a positive number; it is {permits}.");
    }

    /// <summary>Registers the limiter; the limit is read from the host's configuration when the limiter is built.</summary>
    public static void Add(IServiceCollection services)
    {
        services.AddRateLimiter(_ => { });
        services.AddOptions<RateLimiterOptions>().Configure<IConfiguration>((options, configuration) =>
        {
            var permits = PermitsPerMinute(configuration);
            options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
            options.GlobalLimiter = PartitionedRateLimiter.Create<HttpContext, string>(context =>
                permits == 0 || context.Request.Path.Equals(RequestLogLevel.HealthPath, StringComparison.OrdinalIgnoreCase)
                    ? RateLimitPartition.GetNoLimiter("unlimited")
                    : RateLimitPartition.GetFixedWindowLimiter(CallerOf(context), _ => new FixedWindowRateLimiterOptions
                    {
                        PermitLimit = permits,
                        Window = TimeSpan.FromMinutes(1),
                        QueueLimit = 0,
                    }));
            options.OnRejected = async (rejected, cancellationToken) =>
            {
                if (rejected.Lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter))
                    rejected.HttpContext.Response.Headers.RetryAfter =
                        ((int)Math.Ceiling(retryAfter.TotalSeconds)).ToString(CultureInfo.InvariantCulture);
                await rejected.HttpContext.Response.WriteAsJsonAsync(new
                {
                    error = $"Too many requests: this caller may make {permits} a minute ({PermitsKey}).",
                }, cancellationToken).ConfigureAwait(false);
            };
        });
    }
}
