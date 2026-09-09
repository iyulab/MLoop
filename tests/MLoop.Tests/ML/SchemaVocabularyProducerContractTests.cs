using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Tests.ML;

/// <summary>
/// Every value that reaches <see cref="ColumnSchema.DataType"/> must be a name
/// <see cref="SchemaDataTypes"/> defines. Consumers select columns by comparing against that
/// vocabulary, so a producer that writes outside it selects nothing and the run fails a step later
/// on a symptom — a missing feature vector — that names none of the columns responsible.
/// </summary>
/// <remarks>
/// <para>
/// This is a behavioral guard, not a source scan: it runs the producers and inspects what they
/// actually emit. A source scan over <c>DataType = "…"</c> catches only literals, and the paths that
/// worried us here are the ones that <i>compute</i> the value.
/// </para>
/// <para>
/// The producer that made this necessary wrote <c>"Single"</c> for an object-detection bounding-box
/// column while the two columns beside it used the constants — twenty lines under a comment telling
/// the reader to use the vocabulary and not the .NET type name. A guard was the missing half.
/// </para>
/// </remarks>
public class SchemaVocabularyProducerContractTests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { File.Delete(file); } catch { }
        }
    }

    private string WriteCsv(string name, params string[] lines)
    {
        var path = Path.Combine(Path.GetTempPath(), $"mloop-vocab-{Guid.NewGuid()}-{name}");
        File.WriteAllLines(path, lines);
        _tempFiles.Add(path);
        return path;
    }

    [Theory]
    [InlineData("image-classification")]
    [InlineData("object-detection")]
    [InlineData("text-classification")]
    [InlineData(null)]
    public void DirectorySchemaProducer_EmitsOnlyVocabularyNames(string? task)
    {
        var schema = TrainingEngine.BuildDirectoryInputSchema("Label", task, classCount: 3);

        Assert.NotEmpty(schema.Columns);
        Assert.All(schema.Columns, column =>
            Assert.True(
                SchemaDataTypes.IsKnown(column.DataType),
                $"column '{column.Name}' declares DataType '{column.DataType}', which is not one of " +
                string.Join(", ", SchemaDataTypes.All)));
    }

    /// <summary>
    /// The exclusion reasons flow straight into <c>DataType</c> for the columns featurization dropped,
    /// so the reason vocabulary and the DataType vocabulary are the same vocabulary.
    /// </summary>
    [Fact]
    public void ExclusionReasons_AreVocabularyNames()
    {
        // A DateTime column and a constant column: two of the three exclusion reasons, produced by
        // running the real removal chain rather than by asserting the constants against themselves.
        var csv = WriteCsv("exclusions.csv",
            "When,Same,X,Y",
            "2026-01-01,7,1,10",
            "2026-01-02,7,2,20",
            "2026-01-03,7,3,30",
            "2026-01-04,7,4,40");

        var excluded = CsvDataLoader.DetermineExcludedColumns(csv, "Y");

        Assert.NotEmpty(excluded);
        Assert.All(excluded, column =>
            Assert.True(
                SchemaDataTypes.IsKnown(column.Reason),
                $"column '{column.Name}' was excluded with reason '{column.Reason}', which is not one " +
                "of the vocabulary's names and would be written into DataType verbatim"));
    }

    /// <summary>
    /// The companion every scanning guard needs: proof that the predicate above can fail. Without it,
    /// "no offenders" reads the same whether the producers are clean or the check stopped checking.
    /// </summary>
    [Fact]
    public void TheCheckRejectsTheShapeItPolices()
    {
        var offender = new ColumnSchema { Name = "X1", DataType = "Single", Purpose = "Feature" };
        var legitimate = new ColumnSchema { Name = "X2", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" };

        Assert.False(SchemaDataTypes.IsKnown(offender.DataType));
        Assert.True(SchemaDataTypes.IsKnown(legitimate.DataType));
    }
}
