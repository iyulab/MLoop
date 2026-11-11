using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// Puts a NER model's tags back on the words they belong to. ML.NET's NER transformer answers one position to the
/// right: its output starts with the start token's own prediction (always the outside tag), then each word's slot
/// holds the tag of the word before it, and the last word's tag is not written at all. A model that tagged every
/// word right was therefore scored as one that learned nothing — 0.28 of the words by position, 0.82 one position
/// over, 1.00 once the last word is given a successor and the first answer is dropped (measured on a 300-sentence
/// set, 50 held out) — and the quality gate refused it.
/// </summary>
/// <remarks>
/// Applied when a model is loaded to score (<c>ScoringModel.Load</c>) and over a model just trained
/// (the test score), never saved: the model file is the trained model alone, so a model trained before this existed
/// is read right too. In front of the model the sentence gets one more word (<see cref="TextPadding.PadWord"/>) so the last real
/// word has a successor to be read at; behind it the first answer is dropped; the sentence is given back as it was.
/// The sentence column is read from the transformer itself — the column it was trained on.
/// TODO(upstream: https://github.com/dotnet/machinelearning/issues/7889 — the NamedEntityRecognition transformer writes its tags one word to the right) — drop this
/// layer when a release writes one tag per word at its own position.
/// </remarks>
public static class NerAlignment
{
    /// <summary>The ML.NET NER transformer, by type name — MLoop.Core does not reference TorchSharp.</summary>
    public const string TransformerName = "Microsoft.ML.TorchSharp.NasBert.NerTransformer";

    private const string PredictedLabel = "PredictedLabel";

    /// <summary>Whether <paramref name="model"/> (or a transformer in its chain) is a NER model.</summary>
    public static bool IsNer(ITransformer model) => Find(model) is not null;

    /// <summary>The sentence column the NER model reads — the one it was trained on — or <c>null</c> for any other model.</summary>
    public static string? TextColumnOf(ITransformer model)
    {
        if (Find(model) is not { } transformer)
            return null;
        var options = Read(transformer, "Options");
        return options is null ? null : Read(options, "Sentence1ColumnName") as string;
    }

    /// <summary>
    /// <paramref name="model"/> answering one tag per word at the word's own position, or <paramref name="model"/>
    /// itself when it is not a NER model.
    /// </summary>
    public static ITransformer Apply(MLContext mlContext, ITransformer model)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        if (!IsNer(model))
            return model;
        // A NER model whose sentence column cannot be read — the library moved its fields — is not read unaligned in silence: that is
        // the wrong answers this layer exists to prevent.
        var column = TextColumnOf(model)
            ?? throw new InvalidOperationException("The NER model's sentence column could not be read from its transformer (the ML.NET TorchSharp layout changed?); its tags cannot be put back on their words.");
        return Apply(mlContext, model, column);
    }

    /// <summary>The same for a model that reads its sentences from <paramref name="textColumn"/>.</summary>
    public static ITransformer Apply(MLContext mlContext, ITransformer model, string textColumn)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(textColumn);
        return new AlignedModel(new TransformerChain<ITransformer>(
            TextPadding.Mapping(mlContext, textColumn, TextPadding.Padded), model, TagMapping(mlContext), TextPadding.Mapping(mlContext, textColumn, TextPadding.Unpadded)));
    }

    /// <summary>The tags of the real words from the transformer's answer over a <see cref="TextPadding.Padded"/> sentence: all but the start token's.</summary>
    public static string[] Aligned(string[]? raw) => raw is { Length: > 1 } ? raw[1..] : [];

    private static ITransformer? Find(ITransformer? model) => model switch
    {
        null => null,
        _ when model.GetType().FullName == TransformerName => model,
        IEnumerable<ITransformer> chain => chain.Select(Find).FirstOrDefault(t => t is not null),
        _ => null,
    };

    // The options the trainer fitted with live in a non-public field of the transformer's base type.
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

    private static ITransformer TagMapping(MLContext mlContext)
    {
        var input = SchemaDefinition.Create(typeof(Tags));
        input[0].ColumnName = PredictedLabel;
        var output = SchemaDefinition.Create(typeof(Tags));
        output[0].ColumnName = PredictedLabel;
        return mlContext.Transforms.CustomMapping<Tags, Tags>(
                (row, result) => result.Values = Aligned(row.Values), contractName: null, input, output)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<Tags>(), input));
    }

    private sealed class AlignedModel(ITransformer chain) : ITransformer
    {
        public bool IsRowToRowMapper => chain.IsRowToRowMapper;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => chain.GetOutputSchema(inputSchema);

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => chain.GetRowToRowMapper(inputSchema);

        public IDataView Transform(IDataView input) => chain.Transform(input);

        void ICanSaveModel.Save(ModelSaveContext ctx) =>
            throw new NotSupportedException("A NER model is aligned when it is loaded; it is saved as the model alone.");
    }

    private sealed class Tags
    {
        public string[] Values { get; set; } = [];
    }
}
