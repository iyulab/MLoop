using System.Collections.Concurrent;

namespace MLoop.Core.Embedding;

/// <summary>Which encoder made a set of vectors — what a consumer keeps beside them, and re-embeds when it changes.</summary>
/// <param name="Id">The encoder's id.</param>
/// <param name="Version">Its definition's version.</param>
/// <param name="Sha256">Its weights' SHA-256: with <paramref name="Id"/>, the identity of the vector space.</param>
/// <param name="Dimension">The length of each vector.</param>
/// <param name="Normalized">Whether vectors are L2-normalised (always, today): cosine similarity is their dot product.</param>
public sealed record EncoderIdentity(string Id, string Version, string? Sha256, int Dimension, bool Normalized)
{
    /// <summary>The identity of <paramref name="definition"/>'s vectors.</summary>
    public static EncoderIdentity Of(EncoderDefinition definition) =>
        new(definition.Id, definition.Version, definition.Sha256, definition.Dimension, Normalized: true);
}

/// <summary>Vectors with the encoder that made them — the answer of <c>mloop embed --json</c> and <c>POST /embed</c>.</summary>
public sealed record EmbeddingResult(EncoderIdentity Encoder, IReadOnlyList<float[]> Vectors);

/// <summary>
/// Opened encoders, one per installed encoder and its weights, shared by every request. A graph is loaded once; a
/// reinstalled encoder (other weights) opens anew.
/// </summary>
public sealed class EmbedderPool(EncoderStore store) : IDisposable
{
    /// <summary>The most images one request embeds.</summary>
    public const int MaxImages = 64;

    private readonly ConcurrentDictionary<string, Lazy<ImageEmbedder>> _open = new(StringComparer.Ordinal);

    /// <summary>The store encoders are read from.</summary>
    public EncoderStore Store { get; } = store ?? throw new ArgumentNullException(nameof(store));

    /// <summary>
    /// The embedder for <paramref name="id"/> (or the only installed encoder), or why there is none — not installed,
    /// or ONNX Runtime cannot run here.
    /// </summary>
    public (ImageEmbedder? Embedder, string? Refusal) Get(string? id)
    {
        var (encoder, refusal) = Store.Resolve(id);
        if (encoder is null)
            return (null, refusal);

        var key = $"{encoder.Definition.Id}@{encoder.Definition.Sha256}";
        var lazy = _open.GetOrAdd(key, _ => new Lazy<ImageEmbedder>(() => ImageEmbedder.Open(encoder.Definition, encoder.ModelPath)));
        try
        {
            return (lazy.Value, null);
        }
        catch (InvalidOperationException ex)
        {
            _open.TryRemove(key, out _);   // not cached: the runtime may be installed before the next request
            return (null, ex.Message);
        }
    }

    /// <summary>Embeds <paramref name="images"/> with <paramref name="id"/>'s encoder.</summary>
    /// <exception cref="UnreadableImageException">An image could not be decoded.</exception>
    public (EmbeddingResult? Result, string? Refusal) Embed(string? id, IReadOnlyList<byte[]> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count > MaxImages)
            return (null, $"{images.Count} images in one request; at most {MaxImages}.");
        var (embedder, refusal) = Get(id);
        if (embedder is null)
            return (null, refusal);
        return (new EmbeddingResult(EncoderIdentity.Of(embedder.Encoder), embedder.Embed(images)), null);
    }

    public void Dispose()
    {
        foreach (var lazy in _open.Values.Where(l => l.IsValueCreated))
            lazy.Value.Dispose();
        _open.Clear();
    }
}
