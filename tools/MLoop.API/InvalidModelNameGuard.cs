using MLoop.Core.Storage;

namespace MLoop.API;

/// <summary>
/// A model name that is not a valid directory name — <c>../../other</c>, <c>a/b</c> — is the caller's mistake: <c>400</c>.
/// <see cref="ModelName.Resolve"/> is the one door a supplied name walks through and it throws
/// <see cref="InvalidModelNameException"/>; this maps the exception where every endpoint can share it, so a handler that
/// resolves a name from its query or from its body needs nothing of its own. Runs after authentication and
/// authorization: a caller without a valid token is refused as before, whatever name it sends.
/// </summary>
public static class InvalidModelNameGuard
{
    public static IApplicationBuilder UseInvalidModelNameGuard(this IApplicationBuilder app) => app.Use(async (context, next) =>
    {
        try
        {
            await next();
        }
        catch (InvalidModelNameException ex) when (!context.Response.HasStarted)
        {
            context.Response.Clear();
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsJsonAsync(new { error = ex.Message });
        }
    });
}
