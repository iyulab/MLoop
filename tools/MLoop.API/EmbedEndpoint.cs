using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using MLoop.Core.Embedding;

namespace MLoop.API;

/// <summary>
/// <c>POST /embed</c> — images, or texts, to vectors with an installed pretrained encoder (<see cref="EmbedderPool"/>).
/// Takes the images' contents, never a path: a path assumes the client and the server share a disk. <c>?patches=true</c>
/// adds each image's patch vectors and the grid they lie on (at most <see cref="EmbedderPool.MaxPatchImages"/> images).
/// Texts come as <c>{"texts": ["…", …]}</c> and go to a text encoder (at most <see cref="EmbedderPool.MaxTexts"/>).
/// </summary>
public static class EmbedEndpoint
{
    public static void MapEmbed(this WebApplication app)
    {
        app.MapPost("/embed", async (HttpRequest request, [FromQuery] string? encoder, [FromQuery] bool? patches, EmbedderPool pool, CancellationToken ct) =>
        {
            var (texts, images, unreadable) = await ReadInputsAsync(request, ct);
            if (texts is not null)
                return EmbedTexts(pool, encoder, texts, patches == true);
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

    /// <summary>The JSON body's property that carries texts.</summary>
    public const string TextsProperty = "texts";

    private static IResult EmbedTexts(EmbedderPool pool, string? encoder, List<string> texts, bool patches)
    {
        if (patches)
            return Results.BadRequest(new { error = "patches are of images; a text has none" });
        if (texts.Count == 0)
            return Results.BadRequest(new { error = "send {\"texts\": [\"…\", …]} with at least one text" });
        if (EmbedderPool.TooManyTexts(texts.Count) is { } tooMany)
            return Results.Json(new { error = tooMany }, statusCode: StatusCodes.Status413PayloadTooLarge);
        var (embedder, refusal) = pool.GetText(encoder);
        return embedder is null ? Results.Conflict(new { error = refusal }) : Results.Ok(embedder.Embed(texts));
    }

    /// <summary>
    /// The request's texts (a JSON body with <c>texts</c>) or images in order, or why they could not be read. A body with
    /// both is refused: one request is embedded by one encoder.
    /// </summary>
    internal static async Task<(List<string>? Texts, List<byte[]>? Images, string? Error)> ReadInputsAsync(HttpRequest request, CancellationToken ct)
    {
        if (request.HasFormContentType)
            return (null, await ImageInputs.FromFormAsync(await request.ReadFormAsync(ct), ct), null);

        try
        {
            using var body = await JsonDocument.ParseAsync(request.Body, cancellationToken: ct);
            var hasImages = ImageInputs.TryGetImages(body.RootElement, out var images);
            if (TryGetTexts(body.RootElement, out var texts))
            {
                if (hasImages)
                    return (null, null, "send images or texts, not both: one request is embedded by one encoder");
                if (texts.ValueKind != JsonValueKind.Array || texts.EnumerateArray().Any(t => t.ValueKind != JsonValueKind.String))
                    return (null, null, "\"texts\" must be an array of strings");
                return ([.. texts.EnumerateArray().Select(t => t.GetString()!)], null, null);
            }
            if (!hasImages)
                return (null, [], null);
            var (decoded, error) = ImageInputs.FromJson(images);
            return (null, decoded, error);
        }
        catch (JsonException ex)
        {
            return (null, null, $"The body is not JSON of the form {{\"images\": [\"<base64>\", …]}} or {{\"texts\": [\"…\", …]}}: {ex.Message}");
        }
    }

    private static bool TryGetTexts(JsonElement body, out JsonElement texts)
    {
        texts = default;
        if (body.ValueKind != JsonValueKind.Object)
            return false;
        foreach (var property in body.EnumerateObject())
        {
            if (string.Equals(property.Name, TextsProperty, StringComparison.OrdinalIgnoreCase))
            {
                texts = property.Value;
                return true;
            }
        }
        return false;
    }
}
