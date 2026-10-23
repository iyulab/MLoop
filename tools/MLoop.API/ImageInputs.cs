using System.Text.Json;
using MLoop.Core.Embedding;

namespace MLoop.API;

/// <summary>
/// Images sent to the server by their contents — base64 in JSON (<c>{"images": ["…", …]}</c>) or files in a multipart
/// form — the way <c>POST /embed</c> and <c>POST /predict</c> both take them. Never a path: a path assumes the client
/// and the server share a disk.
/// </summary>
public static class ImageInputs
{
    /// <summary>The JSON body's property that carries images.</summary>
    public const string Property = "images";

    /// <summary>Images in one request, at most — <see cref="EmbedderPool.MaxImages"/>, one limit for both endpoints.</summary>
    public const int Max = EmbedderPool.MaxImages;

    /// <summary>What a request with no image is told.</summary>
    public const string Expected = "send {\"images\": [\"<base64>\", …]} or image files as multipart form data";

    /// <summary>A multipart form's files, in order.</summary>
    public static async Task<List<byte[]>> FromFormAsync(IFormCollection form, CancellationToken ct)
    {
        var files = new List<byte[]>(form.Files.Count);
        foreach (var file in form.Files)
        {
            using var buffer = new MemoryStream();
            await file.CopyToAsync(buffer, ct);
            files.Add(buffer.ToArray());
        }
        return files;
    }

    /// <summary>The body's <c>images</c> property, its name compared ignoring case; false when the body has none.</summary>
    public static bool TryGetImages(JsonElement body, out JsonElement images)
    {
        images = default;
        if (body.ValueKind != JsonValueKind.Object)
            return false;
        foreach (var property in body.EnumerateObject())
            if (property.Name.Equals(Property, StringComparison.OrdinalIgnoreCase))
            {
                images = property.Value;
                return true;
            }
        return false;
    }

    /// <summary>The images of a JSON <c>images</c> array of base64 strings, or why they cannot be read.</summary>
    public static (List<byte[]>? Images, string? Error) FromJson(JsonElement images)
    {
        if (images.ValueKind != JsonValueKind.Array)
            return (null, $"\"{Property}\" is not an array of base64 strings.");
        var decoded = new List<byte[]>(images.GetArrayLength());
        var i = 0;
        foreach (var image in images.EnumerateArray())
        {
            try
            {
                decoded.Add(Convert.FromBase64String(image.ValueKind == JsonValueKind.String ? image.GetString()! : "?"));
            }
            catch (FormatException)
            {
                return (null, $"Image {i} is not base64.");
            }
            i++;
        }
        return (decoded, null);
    }

    /// <summary>Why <paramref name="images"/> cannot be taken — none, or more than <see cref="Max"/> (413) — or nothing.</summary>
    public static IResult? Refusal(List<byte[]> images) => images.Count switch
    {
        0 => Results.BadRequest(new { error = $"No image given: {Expected}." }),
        > Max => Results.Json(new { error = $"{images.Count} images in one request; at most {Max}." },
            statusCode: StatusCodes.Status413PayloadTooLarge),
        _ => null,
    };
}
