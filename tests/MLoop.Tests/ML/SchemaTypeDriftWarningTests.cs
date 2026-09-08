using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;

namespace MLoop.Tests.ML;

/// <summary>
/// The saved schema records what each feature column was trained as. When a file's values no longer
/// read as that, the run says which column — before the pipeline fails describing a vector.
/// </summary>
/// <remarks>
/// Schema validation had only ever read the header line: it compared column names and never a
/// value. A column whose contents changed type therefore passed, and the run failed later with
/// <c>Schema mismatch for feature column '__Features__': expected Vector&lt;Single, 3&gt;, got
/// Vector&lt;Single, 2&gt;</c> — a sentence that does not contain the name of the column that
/// changed, which the saved schema had known all along. Warn rather than reject: the loader coerces
/// and the run may still be what the user wants.
/// </remarks>
public class SchemaTypeDriftWarningTests
{
    private static InputSchemaInfo Schema(params (string Name, string Type, string Purpose)[] columns) =>
        new()
        {
            CapturedAt = DateTime.UtcNow,
            Columns = [.. columns.Select(c => new ColumnSchema
            {
                Name = c.Name,
                DataType = c.Type,
                Purpose = c.Purpose,
            })],
        };

    private static string CsvFile(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), $"mloop-schema-{Guid.NewGuid():N}.csv");
        File.WriteAllText(path, content);
        return path;
    }

    private static List<string> WarningsFor(InputSchemaInfo schema, string csv)
    {
        var path = CsvFile(csv);
        try
        {
            var header = File.ReadLines(path).First().Split(',');
            return SchemaValidator.ColumnsWhoseValuesNoLongerReadAsTheirTrainedType(schema, header, path);
        }
        finally
        {
            try { File.Delete(path); } catch (IOException) { }
        }
    }

    [Fact]
    public void ANumericColumnCarryingTextIsNamed()
    {
        var warnings = WarningsFor(
            Schema(("age", SchemaDataTypes.Numeric, "Feature"),
                   ("income", SchemaDataTypes.Numeric, "Feature")),
            "age,income\n30,50000\n41,not-disclosed\n55,61000\n");

        var warning = Assert.Single(warnings);
        Assert.Contains("'income'", warning, StringComparison.Ordinal);
        Assert.Contains("Numeric", warning, StringComparison.Ordinal);
        Assert.Contains("not-disclosed", warning, StringComparison.Ordinal);
        Assert.DoesNotContain("'age'", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileThatStillMatchesSaysNothing()
    {
        Assert.Empty(WarningsFor(
            Schema(("age", SchemaDataTypes.Numeric, "Feature")),
            "age\n30\n41\n55\n"));
    }

    /// <summary>
    /// A blank field is a missing value, which every type admits — reporting it as a type change
    /// would fire on most real files and teach the reader to ignore the warning.
    /// </summary>
    [Fact]
    public void AnEmptyFieldIsNotATypeChange()
    {
        Assert.Empty(WarningsFor(
            Schema(("age", SchemaDataTypes.Numeric, "Feature")),
            "age\n30\n\n55\n"));
    }

    /// <summary>
    /// Only feature columns are checked. The label is the caller's own concern — predict does not
    /// require one, and evaluate handles its type itself.
    /// </summary>
    [Fact]
    public void ALabelColumnIsNotChecked()
    {
        Assert.Empty(WarningsFor(
            Schema(("outcome", SchemaDataTypes.Numeric, "Label")),
            "outcome\nyes\nno\n"));
    }

    [Fact]
    public void ATextColumnAcceptsAnything()
    {
        Assert.Empty(WarningsFor(
            Schema(("note", SchemaDataTypes.Text, "Feature")),
            "note\n123\nhello\n\n"));
    }

    /// <summary>
    /// The parse is the loader's, not the operator's locale — a comma decimal separator is text to
    /// <c>TextLoader</c>, so reporting it as numeric would promise a load that will not happen.
    /// </summary>
    [Theory]
    [InlineData("1.5", true)]
    [InlineData("-2", true)]
    [InlineData("1e3", true)]
    [InlineData("1,5", false)]
    [InlineData("N/A", false)]
    [InlineData("", true)]
    public void NumericReadabilityFollowsTheLoadersParse(string value, bool reads)
    {
        Assert.Equal(reads, SchemaDataTypes.ValueReadsAs(SchemaDataTypes.Numeric, value));
    }
}
