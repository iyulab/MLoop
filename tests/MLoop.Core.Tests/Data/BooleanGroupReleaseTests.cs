using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A categorical feature that holds only yes/no values in the file being predicted or evaluated is
/// not swallowed by the boolean group when training saw it with more values.
/// </summary>
/// <remarks>
/// Measured on a churn model: <c>MultipleLines</c> held No / Yes / No phone service in training; the
/// twelve two-year-contract rows held only No / Yes, the layout inference merged the column into
/// <c>BooleanFeatures</c>, and predict and evaluate both failed with
/// <c>Could not find input column 'MultipleLines'</c> — which <c>evaluate --slice-by</c> turned
/// from rare into routine, because a slice is exactly a subset where a column loses values.
/// </remarks>
public class BooleanGroupReleaseTests
{
    // In this file "lines" reads as boolean; "partner" is boolean in training as well.
    private const string Subset =
        "partner,lines,plan,label\nYes,No,basic,0\nNo,Yes,premium,1\nYes,Yes,basic,0\nNo,No,premium,1\n";

    private static InputSchemaInfo Trained(int linesValues) => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "partner", DataType = SchemaDataTypes.Categorical, Purpose = "Feature", UniqueValueCount = 2 },
            new ColumnSchema { Name = "lines", DataType = SchemaDataTypes.Categorical, Purpose = "Feature", UniqueValueCount = linesValues },
            new ColumnSchema { Name = "plan", DataType = SchemaDataTypes.Categorical, Purpose = "Feature", UniqueValueCount = 2 },
            new ColumnSchema { Name = "label", DataType = SchemaDataTypes.Categorical, Purpose = "Label" },
        ],
    };

    private static (string Name, DataKind Kind, string Ranges)[] Layout(InputSchemaInfo? trainedSchema)
    {
        var path = Path.Combine(Directory.CreateTempSubdirectory("mloop-bool-").FullName, "subset.csv");
        File.WriteAllText(path, Subset);
        var inference = ColumnLayoutInference.Infer(new MLContext(seed: 1), path, "label").Options;

        CsvDataLoader.ReconcileInferredSchemaForInference(inference, trainedSchema, "label", path, preserveColumns: null);

        return [.. inference.Columns!.Select(c =>
            (c.Name, c.DataKind, string.Join(",", (c.Source ?? []).Select(r => $"{r.Min}..{r.Max}"))))];
    }

    [Fact]
    public void AColumnThatHadThreeValuesInTrainingGetsItsOwnTextColumnBack()
    {
        var layout = Layout(Trained(linesValues: 3));

        var lines = Assert.Single(layout, c => c.Name == "lines");
        Assert.Equal(DataKind.String, lines.Kind);
        Assert.Equal("1..1", lines.Ranges);
        // ...and the group no longer reads it, so its width is training's.
        Assert.DoesNotContain(layout, c => c.Kind == DataKind.Boolean && c.Ranges.Contains("1..1"));
        Assert.Contains(layout, c => c.Kind == DataKind.Boolean && c.Ranges == "0..0");
    }

    [Fact]
    public void AColumnThatMayHaveBeenBooleanInTrainingIsLeftInTheGroup()
    {
        var layout = Layout(Trained(linesValues: 2));

        Assert.DoesNotContain(layout, c => c.Name == "lines");
    }

    [Fact]
    public void ThisInferenceDoesGroupTheColumn_WithoutTheSchema()
    {
        // The scan above is only meaningful if the layout inference does swallow the column here.
        var layout = Layout(trainedSchema: null);

        Assert.DoesNotContain(layout, c => c.Name == "lines");
        Assert.Contains(layout, c => c.Kind == DataKind.Boolean && c.Ranges == "0..1");
    }
}
