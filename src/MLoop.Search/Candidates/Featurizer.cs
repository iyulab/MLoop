using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MLoop.Search.Candidates;

/// <summary>How a schema encodes its categorical columns — the second half of a schema, beside the trainer.</summary>
public enum CategoricalEncoding
{
    /// <summary>One slot per category seen in training (<c>OneHotEncoding</c>).</summary>
    OneHot,

    /// <summary>Categories hashed into a fixed number of slots (<c>OneHotHashEncoding</c>).</summary>
    Hash,
}

/// <summary>
/// The feature vector a candidate trains on, built from the column roles as ML.NET AutoML's featurizer
/// built it (<c>AutoCatalog.Featurizer</c>, dotnet/machinelearning <c>cf9093abb6</c>): numbers with
/// missing values replaced, booleans as numbers, categories one-hot or hashed, each text column
/// featurized, all concatenated — text, numbers, categories, booleans, in that order.
/// </summary>
public static class Featurizer
{
    /// <summary>
    /// The featurization of <paramref name="roles"/> over <paramref name="schema"/> into
    /// <paramref name="outputColumn"/>, or <see langword="null"/> when no column is a feature.
    /// </summary>
    public static IEstimator<ITransformer>? Build(
        MLContext mlContext, DataViewSchema schema, ColumnRoles roles, CategoricalEncoding encoding, string outputColumn = "Features")
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentNullException.ThrowIfNull(roles);

        var numericRole = roles.Of(ColumnRole.Numeric).Where(c => c != roles.Label && schema.GetColumnOrNull(c) is not null).ToArray();
        var booleans = numericRole.Where(c => IsBoolean(schema[c].Type)).ToArray();
        var numbers = numericRole.Except(booleans).ToArray();
        var categories = roles.Of(ColumnRole.Categorical).Where(c => c != roles.Label).ToArray();
        var texts = roles.Of(ColumnRole.Text).Where(c => c != roles.Label).ToArray();

        IEstimator<ITransformer>? chain = null;
        void Append(IEstimator<ITransformer> step) => chain = chain is null ? step : chain.Append(step);

        if (numbers.Length > 0)
            Append(mlContext.Transforms.ReplaceMissingValues(Pairs(numbers)));
        if (booleans.Length > 0)
            Append(mlContext.Transforms.Conversion.ConvertType(Pairs(booleans), DataKind.Single));
        if (categories.Length > 0)
        {
            Append(encoding == CategoricalEncoding.OneHot
                ? mlContext.Transforms.Categorical.OneHotEncoding(Pairs(categories))
                : mlContext.Transforms.Categorical.OneHotHashEncoding(Pairs(categories)));
        }
        foreach (var text in texts)
            Append(mlContext.Transforms.Text.FeaturizeText(text, text));

        string[] inputs = [.. texts, .. numbers, .. categories, .. booleans];
        if (inputs.Length == 0)
            return null;
        Append(mlContext.Transforms.Concatenate(outputColumn, inputs));
        return chain;
    }

    private static InputOutputColumnPair[] Pairs(IEnumerable<string> columns) => [.. columns.Select(c => new InputOutputColumnPair(c, c))];

    private static bool IsBoolean(DataViewType type)
        => type is BooleanDataViewType || type is VectorDataViewType { ItemType: BooleanDataViewType };
}
