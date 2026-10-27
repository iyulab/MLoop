using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

public class ServedRowAuditTests
{
    private static readonly InputSchemaInfo Schema = new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new() { Name = "a", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new() { Name = "b", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new() { Name = "kind", DataType = SchemaDataTypes.Categorical, Purpose = "Feature", CategoricalValues = ["x", "y", "z"] },
            new() { Name = "id", DataType = SchemaDataTypes.Numeric, Purpose = "Exclude" },
            new() { Name = "y", DataType = SchemaDataTypes.Categorical, Purpose = "Label", CategoricalValues = ["yes", "no"] },
        ],
    };

    private static Dictionary<string, object> Row(params (string Key, object Value)[] fields) =>
        fields.ToDictionary(f => f.Key, f => f.Value, StringComparer.OrdinalIgnoreCase);

    [Fact]
    public void ACompleteRow_HasNothingToSay()
    {
        var warnings = ServedRowAudit.Describe([Row(("a", 1.5), ("b", 2L), ("kind", "x"))], Schema);

        Assert.Empty(warnings);
    }

    [Fact]
    public void AFeatureTheRowLeavesOut_IsNamedOnce_WithHowManyRowsMissIt()
    {
        var warnings = ServedRowAudit.Describe(
            [Row(("a", 1.0), ("kind", "x")), Row(("a", 2.0), ("b", 3.0), ("kind", "y")), Row(("a", 3.0), ("kind", "z"))], Schema);

        var warning = Assert.Single(warnings);
        Assert.Contains("'b'", warning);
        Assert.Contains("2 of 3 row(s)", warning);
        Assert.Contains("missing", warning);
    }

    [Fact]
    public void ANullOrEmptyValue_CountsAsLeftOut()
    {
        var warnings = ServedRowAudit.Describe([Row(("a", ""), ("b", 1.0), ("kind", "x")), Row(("a", 2.0), ("b", 1.0), ("kind", "x"))], Schema);

        Assert.Contains("Column 'a' is missing from 1 of 2 row(s)", Assert.Single(warnings));
    }

    [Fact]
    public void AValueInANumericColumnThatIsNotANumber_IsNamedWithAnExample()
    {
        var warnings = ServedRowAudit.Describe([Row(("a", "abc"), ("b", 1.0), ("kind", "x"))], Schema);

        var warning = Assert.Single(warnings);
        Assert.Contains("'a'", warning);
        Assert.Contains("not a number", warning);
        Assert.Contains("'abc'", warning);
    }

    [Fact]
    public void ANumberWrittenAsText_IsReadFine_AndTextInACategoricalColumnIsNotAnumberProblem()
    {
        var warnings = ServedRowAudit.Describe([Row(("a", "3.5"), ("b", "1e3"), ("kind", "something new"))], Schema);

        Assert.Empty(warnings);          // an unseen category is the categorical mapper's to report
    }

    [Fact]
    public void TheLabelAndTheColumnsTheModelDoesNotRead_AreNeverAsked()
    {
        var warnings = ServedRowAudit.Describe([Row(("a", 1.0), ("b", 1.0), ("kind", "x"))], Schema);

        Assert.DoesNotContain(warnings, w => w.Contains("'id'") || w.Contains("'y'"));
    }
}
