using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

public class InputSchemaTrainingOnlyTests
{
    private static ColumnSchema Column(string name, string purpose, string type = "Numeric") =>
        new() { Name = name, DataType = type, Purpose = purpose };

    [Fact]
    public void A_training_only_feature_is_recorded_as_not_required()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns = [Column("context", "Feature", "Text"), Column("answer", "Label", "Text"), Column("answer_start", "Feature")]
        };

        var result = schema.WithTrainingOnly(["ANSWER_START"]);

        Assert.Equal(["Feature", "Label", "Ignore"], result.Columns.Select(c => c.Purpose));
        Assert.Equal("Numeric", result.Columns[2].DataType);
        Assert.Equal("Feature", schema.Columns[2].Purpose); // the original is not changed
    }

    [Fact]
    public void Only_a_feature_changes_purpose()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns = [Column("answer", "Label", "Text"), Column("id", "Exclude")]
        };

        var result = schema.WithTrainingOnly(["answer", "id"]);

        Assert.Equal(["Label", "Exclude"], result.Columns.Select(c => c.Purpose));
    }
}
