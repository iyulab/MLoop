using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// Gives a question-answering model its answers whole. ML.NET trains the span of an answer with an inclusive last
/// token and then decodes it as if the last token were exclusive, so every answer comes back without its last token:
/// a single-token answer is the empty string and «Tyrell» is «Ty» (a model that found every span scored
/// <c>exact_match</c> 0 and <c>char_f1</c> 0.26 on 60 held-out questions — 33 answers empty, the other 27 a strict
/// prefix of the right one, none else). The span the model finds is right; the decoder cuts it.
/// </summary>
/// <remarks>
/// The model is taught the span one token longer — <see cref="TrainingAnswer"/> reaches the first character of the
/// token after the answer, so the decoder's cut lands on the answer — and a passage always has a token after its last
/// one (<see cref="TextPadding"/>, at training and when a model is loaded to score: <see cref="Apply(MLContext, ITransformer)"/>).
/// The text of an answer is never changed, only how far the trainer is told it reaches. A model trained before this is
/// read through the same padded passage, harmlessly, and still answers without its last token — retrain it.
/// Measured with the fix on the same set: <c>exact_match</c> 0.93 · <c>char_f1</c> 0.96.
/// TODO(upstream: dotnet/machinelearning — the question-answering decoder reads the inclusive span end as exclusive) — when a
/// release decodes the span as it trains it, drop the extension here and stamp the models trained with it: a model trained
/// this way would answer one token too long.
/// </remarks>
public static class AnswerAlignment
{
    /// <summary>The ML.NET question-answering transformer, by type name — MLoop.Core does not reference TorchSharp.</summary>
    public const string TransformerName = "Microsoft.ML.TorchSharp.Roberta.QATransformer";

    /// <summary>Whether <paramref name="model"/> (or a transformer in its chain) is a question-answering model.</summary>
    public static bool IsQuestionAnswering(ITransformer model) => Find(model) is not null;

    /// <summary>The passage column the model reads — the one it was trained on — or <c>null</c> for any other model.</summary>
    public static string? ContextColumnOf(ITransformer model)
    {
        if (Find(model) is not { } transformer)
            return null;
        var options = Read(transformer, "Options");
        return options is null ? null : Read(options, "ContextColumnName") as string;
    }

    /// <summary><paramref name="model"/> reading its passages with a word after the last one, or itself when it is not a question-answering model.</summary>
    public static ITransformer Apply(MLContext mlContext, ITransformer model)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        if (!IsQuestionAnswering(model))
            return model;
        // See NerAlignment.Apply: an unreadable passage column is a refusal, not an unpadded read.
        var column = ContextColumnOf(model)
            ?? throw new InvalidOperationException("The question-answering model's passage column could not be read from its transformer (the ML.NET TorchSharp layout changed?); its answers cannot be read whole.");
        return Apply(mlContext, model, column);
    }

    /// <summary>The same for a model that reads its passages from <paramref name="contextColumn"/>.</summary>
    public static ITransformer Apply(MLContext mlContext, ITransformer model, string contextColumn)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(contextColumn);
        return new PaddedModel(new TransformerChain<ITransformer>(
            TextPadding.Mapping(mlContext, contextColumn, TextPadding.Padded), model,
            TextPadding.Mapping(mlContext, contextColumn, TextPadding.Unpadded)));
    }

    /// <summary>
    /// What the trainer is told the answer is, for the answer of <paramref name="length"/> characters at
    /// <paramref name="start"/> in <paramref name="paddedContext"/> (a passage <see cref="TextPadding.Padded"/>): the answer
    /// and whatever lies between it and the next token's first character. The trainer finds the token of the last
    /// character, and a space has none — an answer reaching a space is a row it leaves out.
    /// </summary>
    public static string TrainingAnswer(string paddedContext, int start, int length)
    {
        ArgumentNullException.ThrowIfNull(paddedContext);
        var end = start + length;
        while (end < paddedContext.Length - 1 && char.IsWhiteSpace(paddedContext[end]))
            end++;
        return paddedContext.Substring(start, Math.Min(end + 1, paddedContext.Length) - start);
    }

    private static ITransformer? Find(ITransformer? model) => model switch
    {
        null => null,
        _ when model.GetType().FullName == TransformerName => model,
        IEnumerable<ITransformer> chain => chain.Select(Find).FirstOrDefault(t => t is not null),
        _ => null,
    };

    // The options the trainer fitted with live in a non-public field of the transformer.
    private static object? Read(object target, string name)
    {
        const BindingFlags all = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        for (var type = target.GetType(); type is not null; type = type.BaseType)
        {
            if (type.GetField(name, all | BindingFlags.DeclaredOnly) is { } field)
                return field.GetValue(target);
            if (type.GetProperty(name, all | BindingFlags.DeclaredOnly) is { } property)
                return property.GetValue(target);
        }
        return null;
    }

    private sealed class PaddedModel(ITransformer chain) : ITransformer
    {
        public bool IsRowToRowMapper => chain.IsRowToRowMapper;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => chain.GetOutputSchema(inputSchema);

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => chain.GetRowToRowMapper(inputSchema);

        public IDataView Transform(IDataView input) => chain.Transform(input);

        void ICanSaveModel.Save(ModelSaveContext ctx) =>
            throw new NotSupportedException("A question-answering model reads padded passages when it is loaded; it is saved as the model alone.");
    }
}
