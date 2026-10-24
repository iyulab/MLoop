using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using MLoop.Core.Embedding;

namespace MLoop.API;

/// <summary>
/// <c>POST /embed</c> — images to vectors with an installed pretrained encoder (<see cref="EmbedderPool"/>). Takes the
/// images' contents, never a path: a path assumes the client and the server share a disk. <c>?patches=true</c> adds each
/// image's patch vectors and the grid they lie on (at most <see cref="EmbedderPool.MaxPatchImages"/> images).
/// </summary>
public static class EmbedEndpoint
{
    public static void MapEmbed(this WebApplication app)
    {
        app.MapPost("/embed", async (HttpRequest request, [FromQuery] string? encoder, [FromQuery] bool? patches, EmbedderPool pool, CancellationToken ct) =>
        {
            var (images, unreadable) = await ReadImagesAsync(request, ct);
            if (images is null)
                return Results.BadRequest(new { error = unreadable });
            if (ImageInputs.Refusal(images) is { } refused)
                return refused;
            var withPatches = patches == true;
            if (EmbedderPool.TooMany(images.Count, withPatches) is { } tooMany)
                return Results.Json(new { error = tooMany }, statusCode: StatusCodes.Status413PayloadTooLarge);

            var (embedder, refusal) = pool.Get(encoder);
            if (embedder is null)
                return Results.Conflict(new { error = refusal });
            if (withPatches && ImageEmbedder.NoPatches(embedder.Encoder) is { } noPatches)
                return Results.Conflict(new { error = noPatches });

            try
            {
                return Results.Ok(embedder.Embed(images, withPatches));
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
            return (await ImageInputs.FromFormAsync(await request.ReadFormAsync(ct), ct), null);

        try
        {
            using var body = await JsonDocument.ParseAsync(request.Body, cancellationToken: ct);
            return ImageInputs.TryGetImages(body.RootElement, out var images) ? ImageInputs.FromJson(images) : ([], null);
        }
        catch (JsonException ex)
        {
            return (null, $"The body is not JSON of the form {{\"images\": [\"<base64>\", …]}}: {ex.Message}");
        }
    }
}
