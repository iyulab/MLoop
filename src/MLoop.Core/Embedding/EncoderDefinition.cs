namespace MLoop.Core.Embedding;

/// <summary>
/// A pretrained image encoder MLoop can run: which graph, what it expects, and what it returns. An encoder is not an
/// experiment — it is not trained, promoted or gated; its identity for a consumer is <see cref="Id"/> with
/// <see cref="Sha256"/>, the weights the vectors came from.
/// </summary>
/// <param name="Id">The name it is installed and asked for by (<c>dinov2-small</c>).</param>
/// <param name="Version">The catalog's version of the entry; a change of weights is a change of version.</param>
/// <param name="Dimension">The length of one vector.</param>
/// <param name="Preprocessing">How an image becomes the graph's input.</param>
/// <param name="InputName">The graph input that takes pixel values, <c>[N, 3, crop, crop]</c>.</param>
/// <param name="OutputName">The graph output the vector is read from.</param>
/// <param name="Pooling">How a vector is taken from that output.</param>
/// <param name="Url">Where the weights are downloaded from; <see langword="null"/> for an encoder given as a file.</param>
/// <param name="Sha256">The weights' SHA-256, lower-case hex; <see langword="null"/> when not pinned.</param>
public sealed record EncoderDefinition(
    string Id,
    string Version,
    int Dimension,
    ImagePreprocessing Preprocessing,
    string InputName,
    string OutputName,
    EncoderPooling Pooling,
    string? Url = null,
    string? Sha256 = null)
{
    /// <summary>What makes this definition unusable, or <see langword="null"/> when it can run.</summary>
    public string? Problem()
    {
        var p = Preprocessing;
        if (string.IsNullOrWhiteSpace(Id) || string.IsNullOrWhiteSpace(InputName) || string.IsNullOrWhiteSpace(OutputName))
            return "an encoder needs an id, an input name and an output name";
        if (Dimension <= 0)
            return $"dimension {Dimension} is not a vector length";
        if (p is null || p.CropSize <= 0 || p.CropSize > p.ShortestEdge)
            return $"it crops {p?.CropSize} pixels from a shortest edge of {p?.ShortestEdge}";
        if (p.Mean is not { Count: 3 } || p.Std is not { Count: 3 } || p.Std.Any(s => s <= 0))
            return "mean and std need three values (R, G, B), every std above zero";
        return null;
    }
}

/// <summary>
/// An image processor's steps, in order: resize so the shortest edge is <see cref="ShortestEdge"/> (bicubic, as
/// Pillow resamples), crop the centre to <see cref="CropSize"/> square, scale to [0, 1], then subtract
/// <see cref="Mean"/> and divide by <see cref="Std"/> per RGB channel.
/// </summary>
public sealed record ImagePreprocessing(int ShortestEdge, int CropSize, IReadOnlyList<float> Mean, IReadOnlyList<float> Std)
{
    /// <summary>Equal when every step is: the channel values are compared by value, not by list.</summary>
    public bool Equals(ImagePreprocessing? other) =>
        other is not null && ShortestEdge == other.ShortestEdge && CropSize == other.CropSize
        && Mean.SequenceEqual(other.Mean) && Std.SequenceEqual(other.Std);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(ShortestEdge);
        hash.Add(CropSize);
        foreach (var v in Mean.Concat(Std))
            hash.Add(v);
        return hash.ToHashCode();
    }

    /// <summary>ImageNet's channel means and deviations, which most vision encoders are trained with.</summary>
    public static ImagePreprocessing ImageNet(int shortestEdge, int cropSize) =>
        new(shortestEdge, cropSize, [0.485f, 0.456f, 0.406f], [0.229f, 0.224f, 0.225f]);
}

/// <summary>How a vector is read from an encoder's output.</summary>
public enum EncoderPooling
{
    /// <summary>The output is <c>[N, tokens, D]</c>; the vector is the first token's (a transformer's CLS token).</summary>
    FirstToken,

    /// <summary>The output is already <c>[N, D]</c>.</summary>
    None,
}
