using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLoop.Core.Runtime;

namespace MLoop.Core.Embedding;

/// <summary>
/// Runs a pretrained text encoder (<see cref="EncoderDefinition"/> of <see cref="EncoderKind.Text"/>) in ONNX Runtime:
/// texts in, one L2-normalised vector per text out — so the cosine of two vectors is their dot product. Safe to share
/// across threads: the tokenizer and ONNX Runtime's <c>Run</c> are.
/// </summary>
public sealed class TextEmbedder : IDisposable
{
    /// <summary>The graph input that marks which ids are text and which are padding.</summary>
    public const string AttentionMaskName = "attention_mask";

    /// <summary>The graph input a BERT-shaped encoder takes the segment of each id from; all zeros for one text.</summary>
    public const string TokenTypeIdsName = "token_type_ids";

    // Texts are run in batches of this many: a batch is padded to its longest text, so a smaller batch wastes less on
    // padding and holds less of the [N, tokens, D] output at once.
    private const int BatchSize = 32;

    private readonly InferenceSession _session;
    private readonly TextPreprocessor _preprocessor;
    private readonly bool _takesTokenTypes;

    private TextEmbedder(EncoderDefinition encoder, InferenceSession session, TextPreprocessor preprocessor)
    {
        Encoder = encoder;
        _session = session;
        _preprocessor = preprocessor;
        _takesTokenTypes = session.InputMetadata.ContainsKey(TokenTypeIdsName);
    }

    /// <summary>The encoder this runs.</summary>
    public EncoderDefinition Encoder { get; }

    /// <summary>Opens <paramref name="modelPath"/> with the tokenizer at <paramref name="tokenizerPath"/> as <paramref name="encoder"/>.</summary>
    /// <exception cref="InvalidOperationException">
    /// ONNX Runtime cannot run here (the message says how to make it run), the graph is not the encoder described, or the
    /// tokenizer file is not a SentencePiece model.
    /// </exception>
    public static TextEmbedder Open(EncoderDefinition encoder, string modelPath, string tokenizerPath)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        if (encoder.Kind != EncoderKind.Text)
            throw new ArgumentException($"Encoder '{encoder.Id}' reads {encoder.Kind.ToString().ToLowerInvariant()}s, not text.", nameof(encoder));
        if (encoder.Problem() is { } problem)
            throw new ArgumentException($"Encoder '{encoder.Id}' cannot run: {problem}.", nameof(encoder));
        var preprocessor = TextPreprocessor.Open(tokenizerPath, encoder.Text!);
        if (OnnxRuntimeNative.Unavailable() is { } unavailable)
            throw new InvalidOperationException(unavailable);

        using var options = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        var session = new InferenceSession(modelPath, options);
        if (!session.InputMetadata.ContainsKey(encoder.InputName) || !session.InputMetadata.ContainsKey(AttentionMaskName)
            || !session.OutputMetadata.ContainsKey(encoder.OutputName))
        {
            session.Dispose();
            throw new InvalidOperationException(
                $"'{modelPath}' is not encoder '{encoder.Id}': it needs inputs '{encoder.InputName}' and '{AttentionMaskName}' " +
                $"and an output '{encoder.OutputName}'.");
        }
        return new TextEmbedder(encoder, session, preprocessor);
    }

    /// <summary>
    /// One vector per text, in order. <see cref="EmbeddingResult.Truncated"/> counts the texts longer than the encoder
    /// reads (<see cref="TextPreprocessing.MaxTokens"/>): their vectors are of their beginning.
    /// </summary>
    public EmbeddingResult Embed(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var vectors = new float[texts.Count][];
        var truncated = 0;
        for (var start = 0; start < texts.Count; start += BatchSize)
        {
            var count = Math.Min(BatchSize, texts.Count - start);
            var encoded = new (long[] Ids, bool Truncated)[count];
            for (var i = 0; i < count; i++)
            {
                encoded[i] = _preprocessor.Encode(texts[start + i] ?? throw new ArgumentException($"Text {start + i} is null.", nameof(texts)));
                if (encoded[i].Truncated)
                    truncated++;
            }
            Run(encoded, vectors, start);
        }
        return new EmbeddingResult(EncoderIdentity.Of(Encoder), vectors, Truncated: truncated);
    }

    private void Run((long[] Ids, bool Truncated)[] encoded, float[][] vectors, int start)
    {
        var count = encoded.Length;
        var length = encoded.Max(e => e.Ids.Length);
        var ids = new long[count * length];
        var mask = new long[count * length];
        Array.Fill(ids, _preprocessor.PadId);
        for (var i = 0; i < count; i++)
        {
            encoded[i].Ids.CopyTo(ids, i * length);
            Array.Fill(mask, 1L, i * length, encoded[i].Ids.Length);
        }

        int[] shape = [count, length];
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(Encoder.InputName, new DenseTensor<long>(ids, shape)),
            NamedOnnxValue.CreateFromTensor(AttentionMaskName, new DenseTensor<long>(mask, shape)),
        };
        if (_takesTokenTypes)
            inputs.Add(NamedOnnxValue.CreateFromTensor(TokenTypeIdsName, new DenseTensor<long>(new long[count * length], shape)));

        using var results = _session.Run(inputs, [Encoder.OutputName]);
        var output = results[0].AsTensor<float>();
        for (var i = 0; i < count; i++)
            vectors[start + i] = EmbeddingVectors.Normalized(Pool(output, i, encoded[i].Ids.Length));
    }

    private float[] Pool(Tensor<float> output, int row, int tokens)
    {
        var dims = output.Dimensions;
        var rank = Encoder.Pooling == EncoderPooling.None ? 2 : 3;
        if (dims.Length != rank || dims[^1] != Encoder.Dimension)
            throw new InvalidOperationException(
                $"Encoder '{Encoder.Id}' returned [{string.Join(", ", dims.ToArray())}] for '{Encoder.OutputName}'; " +
                $"expected rank {rank} ending in {Encoder.Dimension}.");

        var vector = new float[Encoder.Dimension];
        switch (Encoder.Pooling)
        {
            case EncoderPooling.None:
                for (var d = 0; d < vector.Length; d++)
                    vector[d] = output[row, d];
                break;
            case EncoderPooling.FirstToken:
                for (var d = 0; d < vector.Length; d++)
                    vector[d] = output[row, 0, d];
                break;
            case EncoderPooling.MeanOfTokens:
                // The text's own tokens, markers included — the attention mask's ones; padding is left out.
                for (var t = 0; t < tokens; t++)
                {
                    for (var d = 0; d < vector.Length; d++)
                        vector[d] += output[row, t, d];
                }
                for (var d = 0; d < vector.Length; d++)
                    vector[d] /= tokens;
                break;
        }
        return vector;
    }

    public void Dispose() => _session.Dispose();
}
