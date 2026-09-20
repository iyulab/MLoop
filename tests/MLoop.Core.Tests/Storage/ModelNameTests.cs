using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

/// <summary>
/// The model-name rule, at its home. Three call sites used to carry their own copy and two of them
/// agreed; the third accepted uppercase, underscores and every reserved word while printing a
/// message describing the rule it did not implement.
/// </summary>
public class ModelNameTests
{
    [Theory]
    [InlineData("default", true)]
    [InlineData("my-model", true)]
    [InlineData("churn-2024", true)]
    [InlineData("ab", true)]                  // exactly the minimum
    [InlineData("a", false)]                  // one below it
    [InlineData("MyModel", false)]            // uppercase
    [InlineData("my_model", false)]           // underscore
    [InlineData("_private", false)]           // leading underscore
    [InlineData("123model", false)]           // leading digit
    [InlineData("-model", false)]             // leading hyphen
    [InlineData("model-", false)]             // trailing hyphen
    [InlineData("a--b", false)]               // double hyphen
    [InlineData("my model", false)]           // space
    [InlineData("my.model", false)]           // dot
    [InlineData("", false)]
    [InlineData("   ", false)]
    public void IsValid_AppliesTheStrictRule(string name, bool expected)
        => Assert.Equal(expected, ModelName.IsValid(name));

    [Theory]
    [InlineData("staging")]
    [InlineData("production")]
    [InlineData("temp")]
    [InlineData("cache")]
    [InlineData("index")]
    [InlineData("registry")]
    [InlineData("STAGING")]                   // the reserved check ignores case
    public void IsValid_RejectsReservedNames(string name)
        => Assert.False(ModelName.IsValid(name));

    [Fact]
    public void IsValid_RejectsNamesLongerThanTheMaximum()
    {
        Assert.True(ModelName.IsValid(new string('a', ModelName.MaxLength)));
        Assert.False(ModelName.IsValid(new string('a', ModelName.MaxLength + 1)));
    }

    /// <summary>
    /// The reserved set is the layout's own directory names, read from the layout authority rather
    /// than spelled again here — a name the layout uses inside a model directory cannot also be a
    /// model directory.
    /// </summary>
    [Fact]
    public void ReservedNames_ContainTheLayoutsOwnDirectories()
    {
        Assert.Contains(ExperimentLayout.StagingDirectory, ModelName.ReservedNames);
        Assert.Contains(ExperimentLayout.ProductionDirectory, ModelName.ReservedNames);
    }

    [Fact]
    public void Normalize_TrimsAndLowercases()
        => Assert.Equal("my-model", ModelName.Normalize("  My-Model  "));

    /// <summary>Normalizing is not validating — the two are separate steps for a caller.</summary>
    [Fact]
    public void Normalize_DoesNotMakeAnInvalidNameValid()
    {
        var normalized = ModelName.Normalize(" My_Model ");
        Assert.Equal("my_model", normalized);
        Assert.False(ModelName.IsValid(normalized));
    }

    [Fact]
    public void DescribeViolation_IsNullExactlyWhenTheNameIsValid()
    {
        Assert.Null(ModelName.DescribeViolation("my-model"));
        Assert.NotNull(ModelName.DescribeViolation("My_Model"));
    }

    /// <summary>
    /// A reserved name is well-formed, so the generic "lowercase alphanumeric with hyphens"
    /// sentence describes a rule it already satisfies — the reader is told nothing and reads it as
    /// a bug. The message has to name the actual reason.
    /// </summary>
    [Fact]
    public void DescribeViolation_NamesTheRuleThatWasBroken()
    {
        Assert.Contains("reserved", ModelName.DescribeViolation("staging"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains(ModelName.MaxLength.ToString(), ModelName.DescribeViolation(new string('a', 51))!);
        Assert.Contains(ModelName.MinLength.ToString(), ModelName.DescribeViolation("a")!);
        Assert.Contains("empty", ModelName.DescribeViolation("  "), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The reserved list a message prints is the list the check uses — the prose copy in
    /// init's error text was a fourth place to edit and no compiler watched it.</summary>
    [Fact]
    public void DescribeViolation_ListsEveryReservedName()
    {
        var message = ModelName.DescribeViolation("staging")!;
        foreach (var reserved in ModelName.ReservedNames)
            Assert.Contains(reserved, message, StringComparison.Ordinal);
    }
}
