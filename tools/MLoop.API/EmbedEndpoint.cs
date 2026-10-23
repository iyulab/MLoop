using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using MLoop.Core.Embedding;

namespace MLoop.API;

/// <summary>
/// <c>POST /embed</c> — images to vectors with an installed pretrained encoder (<see cref="EmbedderPool"/>). Takes the
/// images' contents, never a path: a path assumes the client and the server share a disk.
/// </summary>
public static class EmbedEndpoint
{
    /// <summary>The JSON body: images as base64.</summary>
    public sealed record EmbedRequest(IReadOnlyList<string>? Images);

    public static void MapEmbed(this WebApplication app)
    {
        app.MapPost("/embed", async (HttpRequest request, [FromQuery] string? encoder, EmbedderPool pool, CancellationToken ct) =>
        {
            var (images, unreadable) = await ReadImagesAsync(request, ct);
            if (images is null)
                return Results.BadRequest(new { error = unreadable });
            if (images.Count == 0)
                return Results.BadRequest(new { error = "No image given: send {\"images\": [\"<base64>\", …]} or image files as multipart form data." });
            if (images.Count > EmbedderPool.MaxImages)
                return Results.Json(new { error = $"{images.Count} images in one request; at most {EmbedderPool.MaxImages}." },
                    statusCode: StatusCodes.Status413PayloadTooLarge);

            var (embedder, refusal) = pool.Get(encoder);
            if (embedder is null)
                return Results.Conflict(new { error = refusal });

            try
            {
                return Results.Ok(new EmbeddingResult(EncoderIdentity.Of(embedder.Encoder), embedder.Embed(images)));
            }
            catch (UnreadableImageException ex)
            {
                return Results.BadRequest(new { error = ex.Message, index = ex.Index });
            }
        })
        .WithName("Embed")
        .WithTags("Embedding")
        .Produces<EmbeddingResult>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status400BadRequest)
        .Produces<object>(StatusCodes.Status409Conflict)
        .Produces<object>(StatusCodes.Status413PayloadTooLarge)
        .RequireAuthorization();
    }

    /// <summary>The request's images in order, or why they could not be read.</summary>
    internal static async Task<(List<byte[]>? Images, string? Error)> ReadImagesAsync(HttpRequest request, CancellationToken ct)
    {
        if (request.HasFormContentType)
        {
            var form = await request.ReadFormAsync(ct);
            var files = new List<byte[]>(form.Files.Count);
            foreach (var file in form.Files)
            {
                using var buffer = new MemoryStream();
                await file.CopyToAsync(buffer, ct);
                files.Add(buffer.ToArray());
            }
            return (files, null);
        }

        EmbedRequest? body;
        try
        {
            body = await JsonSerializer.DeserializeAsync<EmbedRequest>(request.Body, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }, ct);
        }
        catch (JsonException ex)
        {
            return (null, $"The body is not JSON of the form {{\"images\": [\"<base64>\", …]}}: {ex.Message}");
        }

        var images = new List<byte[]>();
        var encoded = body?.Images ?? [];
        for (var i = 0; i < encoded.Count; i++)
        {
            try
            {
                images.Add(Convert.FromBase64String(encoded[i]));
            }
            catch (FormatException)
            {
                return (null, $"Image {i} is not base64.");
            }
        }
        return (images, null);
    }
}
