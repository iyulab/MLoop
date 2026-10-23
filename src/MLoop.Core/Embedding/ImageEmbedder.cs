using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLoop.Core.Runtime;

namespace MLoop.Core.Embedding;

/// <summary>
/// Runs a pretrained image encoder (<see cref="EncoderDefinition"/>) in ONNX Runtime: images in, one L2-normalised
/// vector per image out — so the cosine of two vectors is their dot product. Safe to share across threads: ONNX
/// Runtime's <c>Run</c> is.
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
        if (encoder.Preprocessing.CropSize > encoder.Preprocessing.ShortestEdge)
            throw new ArgumentException($"Encoder '{encoder.Id}' crops {encoder.Preprocessing.CropSize} pixels from an edge of {encoder.Preprocessing.ShortestEdge}.", nameof(encoder));
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

    /// <summary>One vector per image, in order.</summary>
    /// <exception cref="UnreadableImageException">An image could not be decoded; nothing is returned for the batch.</exception>
    public float[][] Embed(IReadOnlyList<byte[]> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
            return [];

        var p = Encoder.Preprocessing;
        var size = 3 * p.CropSize * p.CropSize;
        var batch = new float[images.Count * size];
        for (var i = 0; i < images.Count; i++)
            ImagePreprocessor.ToTensor(images[i], p, i).CopyTo(batch, i * size);

        var input = new DenseTensor<float>(batch, [images.Count, 3, p.CropSize, p.CropSize]);
        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(Encoder.InputName, input)], [Encoder.OutputName]);
        var output = results[0].AsTensor<float>();
        return [.. Enumerable.Range(0, images.Count).Select(i => Normalized(Pool(output, i)))];
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
