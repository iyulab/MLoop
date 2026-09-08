using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// What the merged feature vector spans is a property of training, not of the file being predicted.
/// </summary>
/// <remarks>
/// <para>
/// <c>InferColumns</c> groups the columns it reads as numeric into one vector and leaves the rest
/// as their own columns. Measured on a three-feature set: training inferred one <c>Features</c>
/// column over source <c>0..2</c>; re-inferring on a prediction file where one of those three
/// carried text gave <c>Features</c> over <c>0..0, 2..2</c> plus a separate <c>String</c> column —
/// the vector two slots wide where the model expects three.
/// </para>
/// <para>
/// Overriding that column's <c>DataKind</c>, which reconciliation already did, cannot fix it: the
/// column is outside the range and setting its kind does not move it back. The run failed inside
/// the pipeline naming a vector width, and the same command with <c>--json</c> — which loads rows
/// rather than inferring a loader — succeeded on the identical input. One command, two answers,
/// decided by an output flag.
/// </para>
/// </remarks>
public class TrainedFeatureRangeTests
{
    private static readonly string[] Features = ["age", "income", "score"];

    private static InputSchemaInfo TrainedOnThreeNumericFeatures() => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            .. Features.Select(name => new ColumnSchema
            {
                Name = name,
                DataType = SchemaDataTypes.Numeric,
                Purpose = "Feature",
            }),
            new ColumnSchema { Name = "label", DataType = SchemaDataTypes.Categorical, Purpose = "Label" },
        ],
    };

    private static InputSchemaInfo TrainedWithIncomeAsText() => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "age", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "income", DataType = SchemaDataTypes.Categorical, Purpose = "Feature" },
            new ColumnSchema { Name = "score", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "label", DataType = SchemaDataTypes.Categorical, Purpose = "Label" },
        ],
    };

    private static string WriteCsv(string name, string content)
    {
        var dir = Directory.CreateTempSubdirectory("mloop-range-").FullName;
        var path = Path.Combine(dir, name);
        File.WriteAllText(path, content);
        return path;
    }

    private static (string Name, string Ranges)[] Layout(string csvPath, InputSchemaInfo? trainedSchema)
    {
        var ml = new MLContext(seed: 1);
        var inference = ml.Auto().InferColumns(csvPath, labelColumnName: "label", groupColumns: true);

        CsvDataLoader.ReconcileInferredSchemaForInference(
            inference, trainedSchema, labelColumn: "label", csvPath, preserveColumns: null);

        return [.. inference.TextLoaderOptions.Columns!.Select(c =>
            (c.Name, string.Join(",", (c.Source ?? []).Select(r => $"{r.Min}..{r.Max}"))))];
    }

    private const string AllNumeric =
        "age,income,score,label\n30,50000,0.1,0\n41,61000,0.2,1\n55,70000,0.3,0\n22,30000,0.9,1\n";

    private const string OneFeatureArrivesNumeric =
        "age,income,score,label\n30,50000,0.1,0\n41,61000,0.2,1\n55,70000,0.3,0\n22,30000,0.9,1\n";

    private const string OneFeatureArrivesAsText =
        "age,income,score,label\n30,not-disclosed,0.1,0\n41,not-disclosed,0.2,1\n55,70000,0.3,0\n22,not-disclosed,0.9,1\n";

    [Fact]
    public void AFeatureThatArrivesAsTextIsPutBackInTheRange()
    {
        var layout = Layout(WriteCsv("textual.csv", OneFeatureArrivesAsText), TrainedOnThreeNumericFeatures());

        var features = Assert.Single(layout, c => c.Name == "Features");
        Assert.Equal("0..2", features.Ranges);
    }

    /// <summary>
    /// And the column it was split into is still there, now overlapping the range. Reading the same
    /// field twice under two names is what a loader is for; removing it would be a claim about who
    /// reads the view.
    /// </summary>
    /// <remarks>
    /// The first version of the repair did remove it, and that is what broke label-less clustering:
    /// its CSV predict path builds the feature vector by concatenating the named columns, so with
    /// the column gone the model's concatenator failed with <c>Could not find input column 'pH'</c>
    /// — a working predict traded for a different failure.
    /// <c>ClusteringFeaturesContractTests</c> is where that is caught end to end; this asserts the
    /// layout decision it rests on.
    /// </remarks>
    [Fact]
    public void TheSplitOutColumnSurvivesAlongsideTheRange()
    {
        var layout = Layout(WriteCsv("textual.csv", OneFeatureArrivesAsText), TrainedOnThreeNumericFeatures());

        Assert.Contains(layout, c => c.Name == "income");
        Assert.Equal("0..2", Assert.Single(layout, c => c.Name == "Features").Ranges);
    }

    /// <summary>
    /// A file that already agrees is left exactly as inference produced it. The repair exists for
    /// the disagreement; applying it unconditionally would make this code the authority on layouts
    /// it has no reason to touch.
    /// </summary>
    [Fact]
    public void AFileThatAlreadyAgreesIsUntouched()
    {
        var path = WriteCsv("numeric.csv", AllNumeric);

        Assert.Equal(Layout(path, trainedSchema: null), Layout(path, TrainedOnThreeNumericFeatures()));
    }

    /// <summary>
    /// The other direction, which the same range decision produces: a feature training read as text
    /// arrives here as digits, so inference pulls it <i>into</i> the vector and the range comes out
    /// one slot too wide. Narrowing it back is only half the repair — the column then has nothing
    /// reading it, and the model asks for it by name.
    /// </summary>
    [Fact]
    public void AFeatureThatArrivesNumericIsPutBackOutsideTheRange()
    {
        var layout = Layout(WriteCsv("numeric.csv", OneFeatureArrivesNumeric), TrainedWithIncomeAsText());

        Assert.Equal("0..0,2..2", Assert.Single(layout, c => c.Name == "Features").Ranges);
        Assert.Equal("1..1", Assert.Single(layout, c => c.Name == "income").Ranges);
    }

    /// <summary>
    /// Its negative control: without the trained schema, inference really does swallow the column,
    /// so the test above is asserting a repair rather than a shape that was already there.
    /// </summary>
    [Fact]
    public void WithoutTheTrainedSchemaTheNumericFeatureStaysSwallowed()
    {
        var layout = Layout(WriteCsv("numeric.csv", OneFeatureArrivesNumeric), trainedSchema: null);

        Assert.Equal("0..2", Assert.Single(layout, c => c.Name == "Features").Ranges);
        Assert.DoesNotContain(layout, c => c.Name == "income");
    }

    /// <summary>
    /// The negative control the other three rest on: without the repair, the range really does lose
    /// a slot. If inference ever stopped splitting the column out, these tests would pass while
    /// asserting nothing.
    /// </summary>
    [Fact]
    public void WithoutTheTrainedSchemaTheRangeLosesASlot()
    {
        var layout = Layout(WriteCsv("textual.csv", OneFeatureArrivesAsText), trainedSchema: null);

        var features = Assert.Single(layout, c => c.Name == "Features");
        Assert.Equal("0..0,2..2", features.Ranges);
        Assert.Contains(layout, c => c.Name == "income");
    }
}
