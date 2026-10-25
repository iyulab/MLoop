using System.Collections.Concurrent;

namespace MLoop.Core.Embedding;

/// <summary>Which encoder made a set of vectors — what a consumer keeps beside them, and re-embeds when it changes.</summary>
/// <param name="Id">The encoder's id.</param>
/// <param name="Version">Its definition's version.</param>
/// <param name="Sha256">Its weights' SHA-256: with <paramref name="Id"/> and <paramref name="Preprocessor"/>, the identity of the vector space.</param>
/// <param name="Dimension">The length of each vector.</param>
/// <param name="Normalized">Whether vectors are L2-normalised (always, today): cosine similarity is their dot product.</param>
/// <param name="Preprocessor">
/// How the input became the encoder's (<see cref="ImagePreprocessor.Revision"/>, <see cref="TextPreprocessor.Revision"/>):
/// the same weights give other vectors when it changes, so vectors kept under one value are not compared with vectors
/// made under another.
/// </param>
/// <param name="Kind">What the encoder reads.</param>
/// <param name="TokenizerSha256">
/// A text encoder's tokenizer model's SHA-256 — other pieces are other ids, so it belongs to the space's identity as the
/// weights do. <see langword="null"/> for an image encoder.
/// </param>
public sealed record EncoderIdentity(
    string Id, string Version, string? Sha256, int Dimension, bool Normalized, string Preprocessor,
    EncoderKind Kind = EncoderKind.Image, string? TokenizerSha256 = null)
{
    /// <summary>The identity of <paramref name="definition"/>'s vectors.</summary>
    public static EncoderIdentity Of(EncoderDefinition definition) => definition.Kind == EncoderKind.Text
        ? new(definition.Id, definition.Version, definition.Sha256, definition.Dimension, Normalized: true,
            TextPreprocessor.Revision(definition.Text!.TokenIds), EncoderKind.Text, definition.TokenizerSha256)
        : new(definition.Id, definition.Version, definition.Sha256, definition.Dimension, Normalized: true, ImagePreprocessor.Revision);
}

/// <summary>One image's patch vectors, row by row over the grid, and the part of the image the grid covers.</summary>
/// <param name="Crop">The centre crop the encoder read, in the image's own pixels; the grid divides it evenly.</param>
/// <param name="Vectors">One L2-normalised vector per patch, <see cref="PatchGrid.Rows"/> × <see cref="PatchGrid.Columns"/>, row by row.</param>
public sealed record ImagePatches(ImageRegion Crop, IReadOnlyList<float[]> Vectors);

/// <summary>
/// Vectors with the encoder that made them — the answer of <c>mloop embed --json</c> and <c>POST /embed</c>. With patches
/// asked for, <see cref="Grid"/> and one <see cref="ImagePatches"/> per image too: a vector per region of the image,
/// for what one vector per image blurs (a small defect against normal variation, where in the image it is). Of texts,
/// <see cref="Truncated"/>: how many were longer than the encoder reads, their vectors being of their beginning.
/// </summary>
public sealed record EmbeddingResult(
    EncoderIdentity Encoder,
    IReadOnlyList<float[]> Vectors,
    PatchGrid? Grid = null,
    IReadOnlyList<ImagePatches>? Patches = null,
    int? Truncated = null);

/// <summary>
/// Opened encoders, one per installed encoder and its weights, shared by every request. A graph is loaded once; a
/// reinstalled encoder (other weights) opens anew.
/// </summary>
public sealed class EmbedderPool(EncoderStore store) : IDisposable
{
    /// <summary>The most images one request embeds.</summary>
    public const int MaxImages = 64;

    /// <summary>The most texts one request embeds.</summary>
    public const int MaxTexts = 256;

    /// <summary>
    /// The most images one request embeds with their patches: a patch grid is hundreds of vectors per image
    /// (dinov2-small: 256 × 384 numbers, ~1 MB of JSON), so a batch of 64 would be a 70 MB answer.
    /// </summary>
    public const int MaxPatchImages = 8;

    private readonly ConcurrentDictionary<string, Lazy<IDisposable>> _open = new(StringComparer.Ordinal);

    private static readonly ConcurrentDictionary<string, EmbedderPool> SharedPools = new(StringComparer.Ordinal);

    /// <summary>
    /// The process's pool over the default store (<see cref="EncoderStore.DefaultRoot"/>, read each time — a test or a
    /// container may move it) — what training and a loaded model that reads encoded text open their encoders from, so
    /// models sharing an encoder share one opened graph. Lives as long as the process.
    /// </summary>
    public static EmbedderPool Shared => SharedPools.GetOrAdd(EncoderStore.DefaultRoot, root => new EmbedderPool(new EncoderStore(root)));

    /// <summary>The store encoders are read from.</summary>
    public EncoderStore Store { get; } = store ?? throw new ArgumentNullException(nameof(store));

    /// <summary>
    /// The image embedder for <paramref name="id"/> (or the only installed image encoder), or why there is none — not
    /// installed, an encoder of text, or ONNX Runtime cannot run here.
    /// </summary>
    public (ImageEmbedder? Embedder, string? Refusal) Get(string? id) =>
        Open(id, EncoderKind.Image, e => ImageEmbedder.Open(e.Definition, e.ModelPath));

    /// <summary>
    /// The text embedder for <paramref name="id"/> (or the only installed text encoder), or why there is none — not
    /// installed, an encoder of images, or ONNX Runtime cannot run here.
    /// </summary>
    public (TextEmbedder? Embedder, string? Refusal) GetText(string? id) =>
        Open(id, EncoderKind.Text, e => TextEmbedder.Open(e.Definition, e.ModelPath, e.TokenizerPath!));

    private (T? Embedder, string? Refusal) Open<T>(string? id, EncoderKind kind, Func<InstalledEncoder, T> open)
        where T : class, IDisposable
    {
        var (encoder, refusal) = Store.Resolve(id, kind);
        if (encoder is null)
            return (null, refusal);

        var key = $"{encoder.Definition.Id}@{encoder.Definition.Sha256}@{encoder.Definition.TokenizerSha256}";
        var lazy = _open.GetOrAdd(key, _ => new Lazy<IDisposable>(() => open(encoder)));
        try
        {
            return ((T)lazy.Value, null);
        }
        catch (InvalidOperationException ex)
        {
            _open.TryRemove(key, out _);   // not cached: the runtime may be installed before the next request
            return (null, ex.Message);
        }
    }

    /// <summary>Why <paramref name="count"/> texts cannot be embedded in one request, or <see langword="null"/>.</summary>
    public static string? TooManyTexts(int count) =>
        count > MaxTexts ? $"{count} texts in one request; at most {MaxTexts}." : null;

    /// <summary>Embeds <paramref name="texts"/> with <paramref name="id"/>'s text encoder.</summary>
    public (EmbeddingResult? Result, string? Refusal) EmbedTexts(string? id, IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (TooManyTexts(texts.Count) is { } tooMany)
            return (null, tooMany);
        var (embedder, refusal) = GetText(id);
        return embedder is null ? (null, refusal) : (embedder.Embed(texts), null);
    }

    /// <summary>Why <paramref name="count"/> images cannot be embedded in one request, or <see langword="null"/>.</summary>
    public static string? TooMany(int count, bool patches) => (patches ? MaxPatchImages : MaxImages) is var max && count > max
        ? $"{count} images in one request{(patches ? " with patches" : "")}; at most {max}."
        : null;

    /// <summary>Embeds <paramref name="images"/> with <paramref name="id"/>'s encoder — with their patches when asked.</summary>
    /// <exception cref="UnreadableImageException">An image could not be decoded.</exception>
    public (EmbeddingResult? Result, string? Refusal) Embed(string? id, IReadOnlyList<byte[]> images, bool patches = false)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (TooMany(images.Count, patches) is { } tooMany)
            return (null, tooMany);
        var (embedder, refusal) = Get(id);
        if (embedder is null)
            return (null, refusal);
        if (patches && ImageEmbedder.NoPatches(embedder.Encoder) is { } noPatches)
            return (null, noPatches);
        return (embedder.Embed(images, patches), null);
    }

    public void Dispose()
    {
        foreach (var lazy in _open.Values.Where(l => l.IsValueCreated))
            lazy.Value.Dispose();
        _open.Clear();
    }
}

/// <summary>Vector arithmetic the embedders share.</summary>
internal static class EmbeddingVectors
{
    /// <summary>Scales <paramref name="vector"/> to length one, in place — a zero vector stays zero.</summary>
    public static float[] Normalized(float[] vector)
    {
        var norm = Math.Sqrt(vector.Sum(v => (double)v * v));
        if (norm > 0)
        {
            for (var d = 0; d < vector.Length; d++)
                vector[d] = (float)(vector[d] / norm);
        }
        return vector;
    }
}
