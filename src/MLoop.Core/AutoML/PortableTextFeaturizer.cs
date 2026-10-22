using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using MLoop.Search;

namespace MLoop.Core.AutoML;

/// <summary>
/// Text featurization that an ONNX runtime computes exactly: counts of character n-grams of
/// lengths 1 to <see cref="NgramLength"/> — used instead of ML.NET's
/// <c>FeaturizeText</c> when a model is trained <c>portable</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>FeaturizeText</c> has no ONNX form. Of the steps that do convert, word tokenization does not
/// reproduce: ONNX Runtime's tokenizer splits differently, and a word-bag model's probabilities
/// drifted by up to 0.04 between ML.NET and the exported graph. <c>NormalizeText</c> does not
/// convert at all. Character n-grams over the raw text matched to 1e-7.
/// </para>
/// <para>
/// Measured on the UCI SMS spam data (4,459 training, 1,115 test messages, the same logistic
/// regression): <c>FeaturizeText</c> accuracy 0.9865, AUC 0.9877; character 1–4-grams accuracy
/// 0.9865, AUC 0.9969; 1–3-grams 0.9857, 0.9970; word bags 0.9686, 0.9778. The featurization
/// is case-sensitive, because lower-casing is one of the steps that does not convert.
/// </para>
/// <para>
/// It is fitted on the training rows only and saved inside the model, in front of the pipeline
/// AutoML picks; the graph's input stays the raw text column.
/// </para>
/// </remarks>
public static class PortableTextFeaturizer
{
    /// <summary>The longest character n-gram; all shorter lengths are included.</summary>
    public const int NgramLength = 4;

    private static string KeysColumn(string textColumn) => textColumn + "__chars";

    /// <summary>
    /// The steps that turn <paramref name="textColumn"/> into a vector of n-gram counts, written
    /// back under the column's own name.
    /// </summary>
    /// <remarks>
    /// In place, so the graph's input and the model's input view keep the column's name.
    /// </remarks>
    public static IEstimator<ITransformer> For(MLContext mlContext, string textColumn)
    {
        // One extractor per length, concatenated — not one extractor with useAllLengths. The latter
        // counts all lengths in ML.NET, but once the model is saved and loaded again ML.NET's ONNX
        // converter writes it as counting only the longest length (min_gram_length = max), so the
        // exported graph silently drops the shorter n-grams: every row of a reloaded model
        // disagreed with ML.NET, while the same transformer exported before saving agreed.
        // TODO(upstream: https://github.com/dotnet/machinelearning/issues/7824) — return to one
        // useAllLengths extractor once a fixed converter ships.
        IEstimator<ITransformer> steps = mlContext.Transforms.Text.TokenizeIntoCharactersAsKeys(KeysColumn(textColumn), textColumn);
        var lengths = Enumerable.Range(1, NgramLength).Select(n => LengthColumn(textColumn, n)).ToArray();
        for (var n = 1; n <= NgramLength; n++)
        {
            steps = steps.Append(mlContext.Transforms.Text.ProduceNgrams(
                LengthColumn(textColumn, n), KeysColumn(textColumn),
                ngramLength: n, useAllLengths: false,
                weighting: NgramExtractingEstimator.WeightingCriteria.Tf));
        }
        return steps.Append(mlContext.Transforms.Concatenate(textColumn, lengths));
    }

    private static string LengthColumn(string textColumn, int length) => $"{textColumn}__{length}grams";

    /// <summary>
    /// Fits the portable featurization of <paramref name="roles"/>' text columns on the training
    /// rows and applies it: each text column becomes a numeric vector under its own name and takes
    /// the numeric role; the intermediate columns are ignored. The search then runs over numeric
    /// data, and <see cref="Prepend"/> puts the fitted featurizer in front of the model it picks.
    /// </summary>
    /// <remarks>
    /// Fitted once, outside the search, rather than handed to it as its pre-featurizer. As a
    /// pre-featurizer it was refitted for every trial and the search came out worse — on the UCI SMS
    /// spam data, test AUC 0.9867 against 0.9952 fitted once (and 0.9874 for the default text
    /// featurization), with the same 40-second budget.
    /// </remarks>
    /// <returns>Unchanged inputs and a null featurizer when there is no text column.</returns>
    public static (ColumnRoles Roles, ITransformer? Featurizer, IDataView Train, IDataView? Validation) FitAndApply(
        MLContext mlContext, ColumnRoles roles, IDataView train, IDataView? validation)
    {
        var textColumns = roles.Of(ColumnRole.Text);
        if (textColumns.Count == 0)
            return (roles, null, train, validation);

        IEstimator<ITransformer>? chain = null;
        foreach (var text in textColumns)
        {
            var steps = For(mlContext, text);
            chain = chain is null ? steps : chain.Append(steps);

            roles = roles.With(text, ColumnRole.Numeric).With(KeysColumn(text), ColumnRole.Ignored);
            for (var n = 1; n <= NgramLength; n++)
                roles = roles.With(LengthColumn(text, n), ColumnRole.Ignored);
        }

        var featurizer = chain!.Fit(train);
        return (roles, featurizer, featurizer.Transform(train), validation is null ? null : featurizer.Transform(validation));
    }

    /// <summary>The model that reads raw text: <paramref name="featurizer"/> (if any) followed by <paramref name="model"/>.</summary>
    public static ITransformer Prepend(ITransformer? featurizer, ITransformer model) =>
        featurizer is null ? model : new TransformerChain<ITransformer>(featurizer, model);
}
