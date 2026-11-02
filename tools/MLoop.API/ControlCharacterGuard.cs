namespace MLoop.API;

/// <summary>
/// A query or path value that holds a control character — a NUL, say — can name no model, experiment, job or date, and the file
/// APIs behind the handlers throw on it ("Null character in path"), which surfaced as a 500 that reads as the server's fault.
/// Refused at the door as the caller's mistake, before any handler runs. Request bodies are not looked at: text there is data,
/// and a tab or a newline in a field is legitimate.
/// </summary>
public static class ControlCharacterGuard
{
    /// <summary>The first control character found in the request's path or query values, or <c>null</c>.</summary>
    public static string? Offender(HttpRequest request)
    {
        // (A NUL in the path never gets this far: the server refuses it before the pipeline runs.)
        if (HasControlCharacter(request.Path.Value))
            return "the path";
        foreach (var (key, value) in request.RouteValues)
        {
            if (value is string text && HasControlCharacter(text))
                return $"the path value '{key}'";
        }
        foreach (var (key, values) in request.Query)
        {
            if (HasControlCharacter(key))
                return $"the query name '{Printable(key)}'";
            foreach (var value in values)
            {
                if (HasControlCharacter(value))
                    return $"the query value of '{key}'";
            }
        }
        return null;
    }

    public static bool HasControlCharacter(string? text) => text is not null && text.Any(char.IsControl);

    private static string Printable(string text) => new(text.Select(c => char.IsControl(c) ? '?' : c).ToArray());

    public static IApplicationBuilder UseControlCharacterGuard(this IApplicationBuilder app) => app.Use(async (context, next) =>
    {
        if (Offender(context.Request) is { } where)
        {
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsJsonAsync(new { error = $"{where} holds a control character, which names nothing." });
            return;
        }
        await next();
    });
}
