using System.Text.Json;

namespace MLoop.API;

/// <summary>
/// A request body the endpoint cannot read is the caller's mistake: <c>400</c>, and the answer says which field. ASP.NET answers
/// a body that is not the endpoint's JSON with a bare <c>400</c> and, outside development, no text at all — a caller who wrote
/// <c>modelName</c> where the model is <c>name</c> (a field the request records refuse: they do not ignore what they do not
/// know) saw only that the request failed. Runs next to <see cref="InvalidModelNameGuard"/>, after authentication and
/// authorization, so a caller without a token is refused as before.
/// </summary>
public static class RequestBodyGuard
{
    public static IApplicationBuilder UseRequestBodyGuard(this IApplicationBuilder app) => app.Use(async (context, next) =>
    {
        try
        {
            await next();
        }
        catch (BadHttpRequestException ex) when (ex.InnerException is JsonException json && !context.Response.HasStarted)
        {
            context.Response.Clear();
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsJsonAsync(new { error = Describe(json) });
        }
    });

    /// <summary>What to tell the caller: the field when the JSON names one the endpoint does not take, else where the body went wrong.</summary>
    public static string Describe(JsonException json)
    {
        var field = json.Path?.Split('.', StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
        // System.Text.Json says «could not be mapped to any .NET member» for a property a type that refuses unknown members does not have.
        if (json.Message.Contains("could not be mapped", StringComparison.Ordinal) && !string.IsNullOrEmpty(field))
            return $"Unknown field '{field}' in the request body: this endpoint does not take it (a field it does not know is refused, not ignored).";
        return string.IsNullOrEmpty(json.Path) || json.Path == "$"
            ? "The request body is not the JSON this endpoint takes."
            : $"The request body is not the JSON this endpoint takes (at {json.Path}).";
    }
}
