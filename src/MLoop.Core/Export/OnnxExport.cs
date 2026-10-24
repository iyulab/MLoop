using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Export;

/// <summary>
/// A trained model that cannot be written as ONNX, with the reason in words a user can act on.
/// Thrown instead of writing a partial graph: a model that converts only in part would score
/// differently from the one that was evaluated.
/// </summary>
public sealed class OnnxExportException : Exception
{
    public OnnxExportException(string message, Exception? inner = null) : base(message, inner) { }
}

/// <summary>One tensor the exported graph reads, and the data columns that fill it.</summary>
/// <param name="Name">The graph input's name.</param>
/// <param name="ElementType">ONNX element type: <c>float32</c>, <c>float64</c>, <c>bool</c>, <c>string</c>, <c>int32</c>, <c>int64</c>, <c>uint32</c>.</param>
/// <param name="Width">Values per row; the tensor's shape is <c>[rows, Width]</c>.</param>
/// <param name="SourceColumns">The data columns that fill the tensor, in slot order.</param>
public sealed record OnnxTensorSpec(string Name, string ElementType, int Width, IReadOnlyList<string> SourceColumns);

/// <summary>One tensor the exported graph writes.</summary>
/// <param name="Name">The graph output's name.</param>
/// <param name="Column">The prediction column it carries (<c>PredictedLabel</c>, <c>Score</c>, <c>Probability</c>).</param>
/// <param name="ElementType">ONNX element type, as in <see cref="OnnxTensorSpec"/>.</param>
/// <param name="Width">Values per row.</param>
public sealed record OnnxOutputSpec(string Name, string Column, string ElementType, int Width);

/// <summary>What <see cref="OnnxExport.Write"/> produced: the graph's inputs and outputs.</summary>
/// <param name="Inputs">Inputs a caller fills from its data.</param>
/// <param name="UnusedInputs">
/// Inputs the graph declares but no output depends on — the label, for a model whose pipeline maps
/// it. A runtime still requires them, so a caller feeds any value of the type.
/// </param>
/// <param name="Outputs">The prediction outputs.</param>
public sealed record OnnxExportResult(
    IReadOnlyList<OnnxTensorSpec> Inputs,
    IReadOnlyList<OnnxTensorSpec> UnusedInputs,
    IReadOnlyList<OnnxOutputSpec> Outputs);

/// <summary>
/// Writes a trained ML.NET model as an ONNX graph that computes the model's prediction columns
/// from the columns of its input view.
/// </summary>
/// <remarks>
/// The input view must be the one the model is scored on — loaded by the same path prediction
/// uses — because the graph's inputs are that view's columns: a merged numeric range arrives as one
/// vector tensor, a text column as a string tensor. <see cref="OnnxTensorSpec.SourceColumns"/>
/// names the data columns behind each so a caller without ML.NET can build the tensors.
/// </remarks>
public static class OnnxExport
{
    /// <summary>The columns an exported graph outputs, when the model produces them.</summary>
    public static readonly IReadOnlyList<string> PredictionColumns = ["PredictedLabel", "Score", "Probability"];

    /// <summary>ONNX names a graph output after the column it carries, with this suffix.</summary>
    public const string OutputSuffix = ".output";

    // Transformers ML.NET cannot convert, by type name, with what the user can do about them.
    // FeaturizeText builds character and word n-grams with TF-IDF weighting in one transformer that
    // has no ONNX form ("ONNX Exportable: No" in ML.NET's transform table); the converter fails on it
    // with an internal KeyNotFoundException that names neither the transform nor the column.
    private static readonly (string TypeName, string Reason)[] NotExportable =
    [
        ("Microsoft.ML.Transforms.Text.TextFeaturizingEstimator+Transformer",
            "it featurizes a string column as free text (ML.NET FeaturizeText), which has no ONNX form"),
    ];

    /// <summary>
    /// Converts <paramref name="model"/> to ONNX and writes it to <paramref name="destination"/>.
    /// </summary>
    /// <param name="mlContext">The context the model was loaded with.</param>
    /// <param name="model">The trained model.</param>
    /// <param name="input">The model's input view (rows are read only for the schema).</param>
    /// <param name="loader">The loader that built <paramref name="input"/>.</param>
    /// <param name="header">The column names the loader's source indices refer to.</param>
    /// <param name="labelColumn">The label column, left out of the graph when no output needs it.</param>
    /// <param name="destination">Where the ONNX bytes go.</param>
    /// <exception cref="OnnxExportException">The model contains a step ONNX cannot express.</exception>
    public static OnnxExportResult Write(
        MLContext mlContext,
        ITransformer model,
        IDataView input,
        TextLoader.Options loader,
        IReadOnlyList<string> header,
        string? labelColumn,
        Stream destination)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(destination);

        if (FindNotExportable(model) is { } reason)
        {
            var stringColumns = input.Schema
                .Where(c => !c.IsHidden && c.Type is TextDataViewType
                            && !string.Equals(c.Name, labelColumn, StringComparison.OrdinalIgnoreCase))
                .Select(c => $"'{c.Name}'")
                .ToList();
            var which = stringColumns.Count > 0
                ? $" Its string columns: {string.Join(", ", stringColumns)}."
                : "";
            throw new OnnxExportException(
                $"This model cannot be exported to ONNX: {reason}.{which} A column that holds a code or a "
                + "category exports once it is declared `type: categorical` under the model's `columns:` in "
                + "mloop.yaml (one-hot encoding) and the model is retrained; a column of free text exports when the "
                + "model is trained portable (`portable: true` in mloop.yaml, or `mloop train --portable`).");
        }

        // The label is a graph input only if the pipeline reads it at scoring time; a model that
        // scores without it does not make a runtime caller invent one. The converter declares the
        // graph's inputs from the data's source, not from a view over it — dropping the label from a
        // view left it a declared input that some runtimes refuse to leave unfed — so the label is
        // left out of the loader itself.
        var view = input;
        var unused = new List<OnnxTensorSpec>();
        if (labelColumn != null && input.Schema.GetColumnOrNull(labelColumn) is { } labelCol)
        {
            var withoutLabel = mlContext.Transforms.DropColumns(labelColumn).Fit(input).Transform(input);
            if (ScoresWithout(model, withoutLabel.Schema))
                view = SourceWithout(mlContext, loader, labelColumn);
            else
                unused.Add(Describe(labelCol, loader, header));
        }

        DataViewSchema scored;
        try
        {
            scored = model.GetOutputSchema(view.Schema);
        }
        catch (Exception ex)
        {
            throw new OnnxExportException(
                "This model cannot be exported to ONNX: the data does not have the columns the model reads — " + ex.Message, ex);
        }

        var outputColumns = PredictionColumns
            .Where(name => scored.GetColumnOrNull(name) is { IsHidden: false })
            .ToArray();
        if (outputColumns.Length == 0)
            throw new OnnxExportException(
                "This model cannot be exported to ONNX: it produces none of the prediction columns "
                + string.Join(", ", PredictionColumns) + ".");

        try
        {
            mlContext.Model.ConvertToOnnx(model, view, destination, outputColumns);
        }
        catch (Exception ex) when (ex is not OnnxExportException)
        {
            throw new OnnxExportException(
                $"This model cannot be exported to ONNX: the converter failed ({ex.GetType().Name}: {ex.Message}).", ex);
        }

        var inputs = view.Schema
            .Where(c => !c.IsHidden && !unused.Any(u => u.Name == c.Name))
            .Select(c => Describe(c, loader, header))
            .ToList();
        var outputs = outputColumns
            .Select(name => scored[name])
            .Select(c => new OnnxOutputSpec(c.Name + OutputSuffix, c.Name, ElementType(c.Type), Width(c.Type)))
            .ToList();

        return new OnnxExportResult(inputs, unused, outputs);
    }

    // An empty view with the loader's columns but one: the schema a graph's inputs are declared from.
    private static IDataView SourceWithout(MLContext mlContext, TextLoader.Options loader, string column)
    {
        var options = new TextLoader.Options();
        foreach (var field in typeof(TextLoader.Options).GetFields(BindingFlags.Public | BindingFlags.Instance))
            field.SetValue(options, field.GetValue(loader));
        options.Columns = [.. loader.Columns.Where(c => !string.Equals(c.Name, column, StringComparison.OrdinalIgnoreCase))];
        return mlContext.Data.CreateTextLoader(options).Load(new MultiFileSource(null));
    }

    /// <summary>
    /// Why <paramref name="model"/> cannot be exported, or <c>null</c> when nothing in it is known
    /// to stop the converter.
    /// </summary>
    public static string? FindNotExportable(ITransformer model)
    {
        foreach (var step in Flatten(model))
        {
            var typeName = step.GetType().FullName;
            foreach (var (name, reason) in NotExportable)
            {
                if (string.Equals(typeName, name, StringComparison.Ordinal))
                    return reason;
            }
        }
        return null;
    }

    private static IEnumerable<ITransformer> Flatten(ITransformer transformer)
    {
        yield return transformer;
        if (transformer is IEnumerable<ITransformer> chain)
        {
            foreach (var inner in chain)
            {
                foreach (var step in Flatten(inner))
                    yield return step;
            }
        }
    }

    private static bool ScoresWithout(ITransformer model, DataViewSchema schema)
    {
        try
        {
            model.GetOutputSchema(schema);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private static OnnxTensorSpec Describe(DataViewSchema.Column column, TextLoader.Options loader, IReadOnlyList<string> header)
    {
        var sources = loader.Columns?
            .FirstOrDefault(c => string.Equals(c.Name, column.Name, StringComparison.Ordinal))?
            .Source?
            .SelectMany(range => Enumerable.Range(range.Min, (range.Max ?? range.Min) - range.Min + 1))
            .Select(index => index >= 0 && index < header.Count ? header[index] : $"#{index}")
            .ToList();

        return new OnnxTensorSpec(column.Name, ElementType(column.Type), Width(column.Type),
            sources is { Count: > 0 } ? sources : [column.Name]);
    }

    private static int Width(DataViewType type) =>
        type is VectorDataViewType vector ? (vector.Size > 0 ? vector.Size : -1) : 1;

    /// <summary>The ONNX element type a column of <paramref name="type"/> becomes.</summary>
    public static string ElementType(DataViewType type)
    {
        var item = type is VectorDataViewType vector ? vector.ItemType : type;
        return item switch
        {
            _ when item == NumberDataViewType.Single => "float32",
            _ when item == NumberDataViewType.Double => "float64",
            _ when item == NumberDataViewType.Int32 => "int32",
            _ when item == NumberDataViewType.Int64 => "int64",
            _ when item == NumberDataViewType.UInt32 => "uint32",
            BooleanDataViewType => "bool",
            TextDataViewType => "string",
            // A key (a class index) is written as its raw unsigned index.
            KeyDataViewType => "uint32",
            _ => item.RawType.Name,
        };
    }
}
