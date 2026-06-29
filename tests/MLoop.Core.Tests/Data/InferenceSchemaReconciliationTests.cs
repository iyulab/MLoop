using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// EVAL-1: pins the shared inference schema-reconciliation contract that predict and evaluate both
/// run after InferColumns. Centralizing it closed the BUG-42 ("String" trained-type) and BUG-16
/// (AllowQuoting) gaps that had been fixed on the predict path but silently missing from evaluate —
/// the F-23/F-26 family of "fix one inference path, the other drifts" defects.
/// </summary>
public class InferenceSchemaReconciliationTests : IDisposable
{
    private readonly string _testDir;
    private readonly MLContext _mlContext;

    public InferenceSchemaReconciliationTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-reconcile-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
        _mlContext = new MLContext(seed: 42);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ReconcileInferredSchema_TrainedSchemaStringType_OverridesInferredKind()
    {
        // BUG-42: a feature training saw as String can be re-inferred as a numeric kind when the
        // inference data is sparse. The trained schema's raw "String" type name must win — this was
        // present in PredictionEngine but missing from EvaluationEngine (only "Categorical"/"Text"
        // mapped to String there), so evaluate could not reproduce the trained feature type.
        var csvPath = WriteCsv("schema_string.csv", "flag,Label\n1.5,10\n2.5,20\n3.5,30\n");
        var inference = _mlContext.Auto().InferColumns(csvPath, labelColumnName: "Label", separatorChar: ',');

        // Sanity: "flag" is inferred as a non-String numeric kind so the override is observable.
        var inferredFlag = inference.TextLoaderOptions.Columns!.First(c => c.Name == "flag").DataKind;
        Assert.NotEqual(DataKind.String, inferredFlag);

        var trainedSchema = MakeSchema(
            ("flag", "String", "Feature"),
            ("Label", "Numeric", "Label"));

        CsvDataLoader.ReconcileInferredSchemaForInference(inference, trainedSchema, "Label", csvPath, null);

        var flagCol = inference.TextLoaderOptions.Columns!.First(c => c.Name == "flag");
        Assert.Equal(DataKind.String, flagCol.DataKind);
    }

    [Fact]
    public void ReconcileInferredSchema_EnablesRfc4180Quoting()
    {
        // BUG-16: quoted fields containing commas (bbox "[1, 2, 3]", attribute dicts) must load as a
        // single column. PredictionEngine set AllowQuoting=true; EvaluationEngine did not, so evaluate
        // mis-split such rows and the feature vector width drifted from training.
        var csvPath = WriteCsv("quoting.csv", "a,Label\n1,2\n");
        var inference = _mlContext.Auto().InferColumns(csvPath, labelColumnName: "Label", separatorChar: ',');

        CsvDataLoader.ReconcileInferredSchemaForInference(inference, null, "Label", csvPath, null);

        Assert.True(inference.TextLoaderOptions.AllowQuoting);
    }

    [Fact]
    public void ReconcileInferredSchema_LeavesLabelColumnUntouched()
    {
        // The label is a legitimate divergence, not drift: predict injects a dummy label / converts
        // Boolean→String, evaluate applies task-specific (multiclass→String, regression→Single, binary
        // string→bool) handling afterward. Reconcile must NOT override the label from the trained
        // schema, or it would clobber that per-engine handling.
        var csvPath = WriteCsv("label_untouched.csv", "a,Label\n1,x\n2,y\n3,x\n");
        var inference = _mlContext.Auto().InferColumns(csvPath, labelColumnName: "Label", separatorChar: ',');
        var labelKindBefore = inference.TextLoaderOptions.Columns!.First(c => c.Name == "Label").DataKind;

        var trainedSchema = MakeSchema(
            ("a", "Numeric", "Feature"),
            ("Label", "Numeric", "Label")); // schema says Single, but reconcile must leave label alone

        CsvDataLoader.ReconcileInferredSchemaForInference(inference, trainedSchema, "Label", csvPath, null);

        var labelKindAfter = inference.TextLoaderOptions.Columns!.First(c => c.Name == "Label").DataKind;
        Assert.Equal(labelKindBefore, labelKindAfter);
    }

    [Fact]
    public void ReconcileInferredSchema_PreservesGroupColumn_SplitsOutOfMergedRange()
    {
        // F-23/F-26: a numeric group/user/item column adjacent to features gets merged into the
        // Features range by InferColumns. Reconcile must split it back out so a model's key transform
        // can address it. This pins that reconcile threads preserveColumns into ApplyColumnPreservation.
        var csvPath = WriteCsv("preserve.csv", "QueryId,F1,F2,Label\n0,0.1,0.2,1\n0,0.3,0.4,2\n1,0.5,0.6,3\n");
        var inference = _mlContext.Auto().InferColumns(csvPath, labelColumnName: "Label", separatorChar: ',');

        CsvDataLoader.ReconcileInferredSchemaForInference(
            inference, null, "Label", csvPath, new[] { "QueryId" });

        // QueryId must exist as an individually-addressable single-source column.
        var queryCol = inference.TextLoaderOptions.Columns!
            .FirstOrDefault(c => c.Name.Equals("QueryId", StringComparison.OrdinalIgnoreCase));
        Assert.NotNull(queryCol);
        Assert.Single(queryCol!.Source);
    }

    private string WriteCsv(string name, string content)
    {
        var path = Path.Combine(_testDir, name);
        File.WriteAllText(path, content, new System.Text.UTF8Encoding(true));
        return path;
    }

    private static InputSchemaInfo MakeSchema(params (string Name, string DataType, string Purpose)[] columns)
    {
        return new InputSchemaInfo
        {
            Columns = columns.Select(c => new ColumnSchema
            {
                Name = c.Name,
                DataType = c.DataType,
                Purpose = c.Purpose,
            }).ToList(),
            CapturedAt = DateTime.UtcNow,
        };
    }
}
