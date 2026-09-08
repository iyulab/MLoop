using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// Which original value <c>false</c> stands for is one decision, made in one place. Training reads
/// it to convert the label; prediction reads it to say what it predicted.
/// </summary>
public class BinaryLabelVocabularyTests
{
    [Fact]
    public void TheNegativeClassIsTheFirstValueCaseInsensitively()
    {
        Assert.Equal(("NG", "OK"), BinaryLabelVocabulary.Order(["OK", "NG"]));
        Assert.Equal(("Fail", "Pass"), BinaryLabelVocabulary.Order(["Pass", "Fail"]));
        Assert.Equal(("0", "1"), BinaryLabelVocabulary.Order(["1", "0"]));
    }

    /// <summary>
    /// The reason this is a shared function rather than a rule each caller restates: the two
    /// orderings genuinely disagree. The schema's own <c>CategoricalValues</c> list is sorted
    /// ordinally, where an upper-case letter precedes every lower-case one — so reading position 0
    /// out of that list would name the wrong class for a vocabulary like this one, silently
    /// inverting every prediction the user reads.
    /// </summary>
    [Fact]
    public void ItDisagreesWithAnOrdinalSortAndThatIsTheWholePoint()
    {
        string[] values = ["a", "B"];

        Assert.Equal(("a", "B"), BinaryLabelVocabulary.Order(values));
        Assert.Equal(["B", "a"], values.OrderBy(v => v, StringComparer.Ordinal));
    }

    public static TheoryData<string[]> NonBinaryVocabularies => new()
    {
        { new[] { "OK" } },
        { new[] { "low", "mid", "high" } },
        { Array.Empty<string>() },
    };

    [Theory]
    [MemberData(nameof(NonBinaryVocabularies))]
    public void ANonBinaryVocabularyResolvesToNothing(string[] values)
    {
        Assert.Null(BinaryLabelVocabulary.Order(values));
    }

    [Fact]
    public void BlankValuesAndDuplicatesDoNotCountAsClasses()
    {
        Assert.Equal(("NG", "OK"), BinaryLabelVocabulary.Order(["OK", " ", "NG", "OK", ""]));
    }

    /// <summary>
    /// Without a recorded vocabulary — an older experiment, or a label already Boolean in the data
    /// — both prediction paths still have to render the same string as each other. They fall back
    /// to the Boolean's own names rather than to whatever each writer happens to print.
    /// </summary>
    [Fact]
    public void WithoutAVocabularyBothSidesStillGetOneAnswer()
    {
        Assert.Equal("True", BinaryLabelVocabulary.Render(true, null));
        Assert.Equal("False", BinaryLabelVocabulary.Render(false, null));
    }

    [Fact]
    public void TheLabelColumnIsWhereTheVocabularyIsRead()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns =
            [
                new ColumnSchema
                {
                    Name = "grade", DataType = SchemaDataTypes.Categorical, Purpose = "Feature",
                    CategoricalValues = ["x", "y"],
                },
                new ColumnSchema
                {
                    Name = "result", DataType = SchemaDataTypes.Categorical, Purpose = "Label",
                    CategoricalValues = ["NG", "OK"],
                },
            ],
        };

        Assert.Equal(("NG", "OK"), BinaryLabelVocabulary.Of(schema));
        Assert.Equal("OK", BinaryLabelVocabulary.Render(true, BinaryLabelVocabulary.Of(schema)));
    }
}
