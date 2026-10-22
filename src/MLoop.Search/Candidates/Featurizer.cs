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
/// The feature vector a candidate trains on, built from the column roles in the shape of ML.NET AutoML's
/// featurizer (<c>AutoCatalog.Featurizer</c>, dotnet/machinelearning <c>cf9093abb6</c>): numbers with
/// missing values replaced, booleans as numbers, categories one-hot or hashed, each text column
/// featurized, all concatenated — text, numbers, missing-value indicators, categories, booleans, in that order.
/// </summary>
/// <remarks>
/// A missing number is replaced by its column's mean (per slot of a vector), fitted with the rest of the
/// pipeline on the rows it trains on — not by zero, as AutoML did, which reads as a real value wherever
/// zero lies inside a column's range. A column that held missing values in training also gets an
/// indicator (1 where the value was missing), so a model can still tell the replacement from a real mean
/// and learn from the missingness itself.
/// </remarks>
public static class Featurizer
{
    /// <summary>
    /// The featurization of <paramref name="roles"/> over <paramref name="schema"/> into
    /// <paramref name="outputColumn"/>, or <see langword="null"/> when no column is a feature.
    /// </summary>
    /// <param name="mlContext">The context the estimators are made in.</param>
    /// <param name="schema">The schema of the rows the featurization will be fitted on.</param>
    /// <param name="roles">Each column's role.</param>
    /// <param name="encoding">How categorical columns are encoded.</param>
    /// <param name="withMissingValues">
    /// The numeric columns that held missing values in training (<see cref="ColumnsWithMissingValues"/>) —
    /// each gets an indicator. Other numeric columns are mean-replaced alone, so a column that is never
    /// missing adds no constant feature.
    /// </param>
    /// <param name="outputColumn">The name of the feature vector.</param>
    public static IEstimator<ITransformer>? Build(
        MLContext mlContext, DataViewSchema schema, ColumnRoles roles, CategoricalEncoding encoding,
        IReadOnlySet<string>? withMissingValues = null, string outputColumn = "Features")
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentNullException.ThrowIfNull(roles);

        var (numbers, booleans) = Numeric(schema, roles);
        var indicated = numbers.Where(c => withMissingValues?.Contains(c) == true)
            .Select(c => (Column: c, Indicator: IndicatorName(schema, c))).ToArray();
        var categories = roles.Of(ColumnRole.Categorical).Where(c => c != roles.Label).ToArray();
        var texts = roles.Of(ColumnRole.Text).Where(c => c != roles.Label).ToArray();

        IEstimator<ITransformer>? chain = null;
        void Append(IEstimator<ITransformer> step) => chain = chain is null ? step : chain.Append(step);

        if (indicated.Length > 0)
            Append(mlContext.Transforms.IndicateMissingValues([.. indicated.Select(i => new InputOutputColumnPair(i.Indicator, i.Column))]));
        if (numbers.Length > 0)
            Append(mlContext.Transforms.ReplaceMissingValues(Pairs(numbers), MissingValueReplacingEstimator.ReplacementMode.Mean));
        if (indicated.Length > 0)
            Append(mlContext.Transforms.Conversion.ConvertType(Pairs(indicated.Select(i => i.Indicator)), DataKind.Single));
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

        string[] inputs = [.. texts, .. numbers, .. indicated.Select(i => i.Indicator), .. categories, .. booleans];
        if (inputs.Length == 0)
            return null;
        Append(mlContext.Transforms.Concatenate(outputColumn, inputs));
        return chain;
    }

    /// <summary>
    /// The numeric feature columns of <paramref name="roles"/> that hold at least one missing value in
    /// <paramref name="data"/> — read once over the training partition, before any fold is cut, so every
    /// trial and the refit build the same features.
    /// </summary>
    public static IReadOnlySet<string> ColumnsWithMissingValues(IDataView data, ColumnRoles roles)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(roles);

        var columns = Numeric(data.Schema, roles).Numbers
            .Select(name => data.Schema[name])
            .Where(c => c.Type is NumberDataViewType or VectorDataViewType { ItemType: NumberDataViewType })
            .ToArray();
        var found = new HashSet<string>(StringComparer.Ordinal);
        if (columns.Length == 0)
            return found;

        var checks = new List<(string Name, Func<bool> HasMissing)>();
        using var cursor = data.GetRowCursor(columns);
        foreach (var column in columns)
            checks.Add((column.Name, MissingCheck(cursor, column)));
        while (cursor.MoveNext() && found.Count < columns.Length)
        {
            foreach (var (name, hasMissing) in checks)
            {
                if (!found.Contains(name) && hasMissing())
                    found.Add(name);
            }
        }
        return found;
    }

    private static Func<bool> MissingCheck(DataViewRowCursor cursor, DataViewSchema.Column column)
    {
        var item = column.Type is VectorDataViewType vector ? vector.ItemType : column.Type;
        if (column.Type is VectorDataViewType)
        {
            if (item == NumberDataViewType.Single)
            {
                var get = cursor.GetGetter<VBuffer<float>>(column);
                VBuffer<float> value = default;
                return () => { get(ref value); foreach (var x in value.GetValues()) if (float.IsNaN(x)) return true; return false; };
            }
            if (item == NumberDataViewType.Double)
            {
                var get = cursor.GetGetter<VBuffer<double>>(column);
                VBuffer<double> value = default;
                return () => { get(ref value); foreach (var x in value.GetValues()) if (double.IsNaN(x)) return true; return false; };
            }
        }
        else if (item == NumberDataViewType.Single)
        {
            var get = cursor.GetGetter<float>(column);
            float value = default;
            return () => { get(ref value); return float.IsNaN(value); };
        }
        else if (item == NumberDataViewType.Double)
        {
            var get = cursor.GetGetter<double>(column);
            double value = default;
            return () => { get(ref value); return double.IsNaN(value); };
        }
        return () => false;   // an integer type has no missing value
    }

    private static (string[] Numbers, string[] Booleans) Numeric(DataViewSchema schema, ColumnRoles roles)
    {
        var numericRole = roles.Of(ColumnRole.Numeric).Where(c => c != roles.Label && schema.GetColumnOrNull(c) is not null).ToArray();
        var booleans = numericRole.Where(c => IsBoolean(schema[c].Type)).ToArray();
        return (numericRole.Except(booleans).ToArray(), booleans);
    }

    // "<column>.missing", or with a number added when the data already has a column of that name.
    private static string IndicatorName(DataViewSchema schema, string column)
    {
        var name = $"{column}.missing";
        for (var n = 2; schema.GetColumnOrNull(name) is not null; n++)
            name = $"{column}.missing{n}";
        return name;
    }

    private static InputOutputColumnPair[] Pairs(IEnumerable<string> columns) => [.. columns.Select(c => new InputOutputColumnPair(c, c))];

    private static bool IsBoolean(DataViewType type)
        => type is BooleanDataViewType || type is VectorDataViewType { ItemType: BooleanDataViewType };
}
