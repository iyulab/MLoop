using Microsoft.ML.OnnxRuntime;
using MLoop.Core.Runtime;

namespace MLoop.Core.Embedding;

/// <summary>What looking at a graph found: what is wrong with it as the encoder described, or why it was not looked at.</summary>
/// <param name="Problem">What makes the graph not the encoder its definition describes; <see langword="null"/> when nothing does.</param>
/// <param name="NotChecked">Why the graph could not be opened to look (ONNX Runtime is not installed, say); <see langword="null"/> when it was.</param>
public sealed record GraphCheckResult(string? Problem, string? NotChecked)
{
    /// <summary>Whether the graph was opened and matches its definition.</summary>
    public bool Passed => Problem is null && NotChecked is null;
}

/// <summary>
/// Checks, when an encoder is added, what <see cref="ImageEmbedder"/> and <see cref="TextEmbedder"/> would otherwise find
/// at the first use: that the graph has the input and output the definition names, and that the output has the rank and
/// width its pooling and dimension say. A definition that cannot run is refused when added (<see cref="EncoderDefinition.Problem"/>);
/// this is the same for what only the graph can tell.
/// </summary>
public static class EncoderGraphCheck
{
    private const string AttentionMask = "attention_mask";

    /// <summary>Opens <paramref name="modelPath"/> and compares it with <paramref name="definition"/> (which must itself be able to run).</summary>
    public static GraphCheckResult Check(EncoderDefinition definition, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (OnnxRuntimeNative.Unavailable() is { } unavailable)
            return new GraphCheckResult(null, unavailable);

        try
        {
            using var options = OnnxSessionOptions.Create();
            using var session = new InferenceSession(modelPath, options);
            return new GraphCheckResult(Compare(definition, session), null);
        }
        catch (OnnxRuntimeException ex)
        {
            return new GraphCheckResult($"'{modelPath}' is not a graph ONNX Runtime can open: {ex.Message}", null);
        }
    }

    private static string? Compare(EncoderDefinition definition, InferenceSession session)
    {
        var inputs = session.InputMetadata;
        var outputs = session.OutputMetadata;
        if (!inputs.ContainsKey(definition.InputName))
            return $"the graph has no input '{definition.InputName}' (its inputs: {string.Join(", ", inputs.Keys)})";
        if (definition.Kind == EncoderKind.Text && !inputs.ContainsKey(AttentionMask))
            return $"a text encoder's graph takes '{AttentionMask}' beside '{definition.InputName}' (its inputs: {string.Join(", ", inputs.Keys)})";
        if (!outputs.TryGetValue(definition.OutputName, out var output))
            return $"the graph has no output '{definition.OutputName}' (its outputs: {string.Join(", ", outputs.Keys)})";

        var rank = definition.Pooling == EncoderPooling.None ? 2 : 3;
        var dims = output.Dimensions;
        if (dims.Length == 0)
            return null;   // a graph that does not say its output's rank: the first run will
        if (dims.Length != rank)
            return $"output '{definition.OutputName}' is [{string.Join(", ", dims.Select(Describe))}], rank {dims.Length}, but pooling " +
                   $"'{PoolingWord(definition.Pooling)}' reads rank {rank} ({(rank == 3 ? "[images, tokens, dimension]" : "[images, dimension]")})";
        if (dims[^1] > 0 && dims[^1] != definition.Dimension)
            return $"output '{definition.OutputName}' is {dims[^1]} wide, but the definition says dimension {definition.Dimension}";
        return null;
    }

    private static string Describe(int dimension) => dimension > 0 ? dimension.ToString() : "?";

    private static string PoolingWord(EncoderPooling pooling) => pooling switch
    {
        EncoderPooling.FirstToken => "first-token",
        EncoderPooling.MeanOfTokens => "mean-of-tokens",
        _ => "none",
    };
}
