using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLoop.Core.Runtime;

namespace MLoop.Core.Embedding;

/// <summary>
/// Runs a pretrained image encoder (<see cref="EncoderDefinition"/>) in ONNX Runtime: images in, one L2-normalised
/// vector per image out — so the cosine of two vectors is their dot product — and, asked for, one per patch of the
/// image from the same run. Safe to share across threads: ONNX Runtime's <c>Run</c> is.
/// </summary>
public sealed class ImageEmbedder : IDisposable
{
    private readonly InferenceSession _session;

    private ImageEmbedder(EncoderDefinition encoder, InferenceSession session)
    {
        Encoder = encoder;
        _session = session;
    }

    /// <summary>The encoder this runs.</summary>
    public EncoderDefinition Encoder { get; }

    /// <summary>Opens <paramref name="modelPath"/> as <paramref name="encoder"/>.</summary>
    /// <exception cref="InvalidOperationException">ONNX Runtime cannot run here (the message says how to make it run), or the graph is not the encoder described.</exception>
    public static ImageEmbedder Open(EncoderDefinition encoder, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        if (encoder.Problem() is { } problem)
            throw new ArgumentException($"Encoder '{encoder.Id}' cannot run: {problem}.", nameof(encoder));
        if (OnnxRuntimeNative.Unavailable() is { } unavailable)
            throw new InvalidOperationException(unavailable);

        using var options = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        var session = new InferenceSession(modelPath, options);
        if (!session.InputMetadata.ContainsKey(encoder.InputName) || !session.OutputMetadata.ContainsKey(encoder.OutputName))
        {
            session.Dispose();
            throw new InvalidOperationException(
                $"'{modelPath}' is not encoder '{encoder.Id}': it has no input '{encoder.InputName}' or no output '{encoder.OutputName}'.");
        }
        return new ImageEmbedder(encoder, session);
    }

    /// <summary>
    /// Why <paramref name="encoder"/> gives no patch vectors, or <see langword="null"/> when it does.
    /// </summary>
    public static string? NoPatches(EncoderDefinition encoder) => encoder.Grid is null
        ? $"Encoder '{encoder.Id}' does not say how its output tiles the image (\"patchSize\" in its encoder.json), so it gives no patch vectors."
        : null;

    /// <summary>One vector per image, in order — and with <paramref name="patches"/>, every image's patch vectors.</summary>
    /// <exception cref="UnreadableImageException">An image could not be decoded; nothing is returned for the batch.</exception>
    /// <exception cref="InvalidOperationException">Patches were asked of an encoder that gives none (<see cref="NoPatches"/>).</exception>
    public EmbeddingResult Embed(IReadOnlyList<byte[]> images, bool patches = false)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (patches && NoPatches(Encoder) is { } noPatches)
            throw new InvalidOperationException(noPatches);
        var identity = EncoderIdentity.Of(Encoder);
        if (images.Count == 0)
            return patches ? new EmbeddingResult(identity, [], Encoder.Grid, []) : new EmbeddingResult(identity, []);

        var p = Encoder.Preprocessing;
        var size = 3 * p.CropSize * p.CropSize;
        var batch = new float[images.Count * size];
        var crops = new ImageRegion[images.Count];
        for (var i = 0; i < images.Count; i++)
        {
            var (tensor, crop) = ImagePreprocessor.Prepare(images[i], p, i);
            tensor.CopyTo(batch, i * size);
            crops[i] = crop;
        }

        var input = new DenseTensor<float>(batch, [images.Count, 3, p.CropSize, p.CropSize]);
        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(Encoder.InputName, input)], [Encoder.OutputName]);
        var output = results[0].AsTensor<float>();
        float[][] vectors = [.. Enumerable.Range(0, images.Count).Select(i => Normalized(Pool(output, i)))];
        if (!patches)
            return new EmbeddingResult(identity, vectors);

        var grid = Encoder.Grid!;
        return new EmbeddingResult(identity, vectors, grid,
            [.. Enumerable.Range(0, images.Count).Select(i => new ImagePatches(crops[i], Patches(output, i, grid)))]);
    }

    private float[] Pool(Tensor<float> output, int row)
    {
        var dims = output.Dimensions;
        var (rank, width) = Encoder.Pooling switch
        {
            EncoderPooling.FirstToken => (3, dims.Length == 3 ? dims[2] : -1),
            _ => (2, dims.Length == 2 ? dims[1] : -1),
        };
        if (dims.Length != rank || width != Encoder.Dimension)
            throw new InvalidOperationException(
                $"Encoder '{Encoder.Id}' returned [{string.Join(", ", dims.ToArray())}] for '{Encoder.OutputName}'; " +
                $"expected rank {rank} ending in {Encoder.Dimension}.");

        var vector = new float[width];
        for (var d = 0; d < width; d++)
            vector[d] = rank == 3 ? output[row, 0, d] : output[row, d];
        return vector;
    }

    // The patch tokens are the output's last rows × columns tokens, row by row: a vision transformer puts its class token
    // (and any register tokens) first.
    private float[][] Patches(Tensor<float> output, int row, PatchGrid grid)
    {
        var tokens = output.Dimensions[1];
        var count = grid.Rows * grid.Columns;
        if (tokens < count + 1)
            throw new InvalidOperationException(
                $"Encoder '{Encoder.Id}' returned {tokens} tokens for '{Encoder.OutputName}'; a {grid.Rows} × {grid.Columns} " +
                $"grid of {grid.PatchSize}-pixel patches needs {count} after the class token.");

        var first = tokens - count;
        var patches = new float[count][];
        for (var t = 0; t < count; t++)
        {
            var vector = new float[Encoder.Dimension];
            for (var d = 0; d < vector.Length; d++)
                vector[d] = output[row, first + t, d];
            patches[t] = Normalized(vector);
        }
        return patches;
    }

    private static float[] Normalized(float[] vector)
    {
        var norm = Math.Sqrt(vector.Sum(v => (double)v * v));
        if (norm > 0)
        {
            for (var d = 0; d < vector.Length; d++)
                vector[d] = (float)(vector[d] / norm);
        }
        return vector;
    }

    public void Dispose() => _session.Dispose();
}
