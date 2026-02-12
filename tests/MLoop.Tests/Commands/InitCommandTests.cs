using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class InitCommandTests
{
    // --- IsValidModelName tests ---

    [Theory]
    [InlineData("default", true)]
    [InlineData("my-model", true)]
    [InlineData("churn-predictor", true)]
    [InlineData("ab", true)]   // minimum 2 chars
    [InlineData("a", false)]   // too short
    [InlineData("", false)]
    [InlineData("  ", false)]
    [InlineData("MyModel", false)]   // uppercase not allowed
    [InlineData("my_model", false)]  // underscore not allowed
    [InlineData("123model", false)]  // must start with letter
    [InlineData("-model", false)]    // must start with letter
    [InlineData("model-", false)]    // can't end with hyphen
    public void IsValidModelName_ValidatesCorrectly(string name, bool expected)
    {
        var result = InitCommand.IsValidModelName(name);
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("staging")]
    [InlineData("production")]
    [InlineData("temp")]
    [InlineData("cache")]
    [InlineData("index")]
    [InlineData("registry")]
    public void IsValidModelName_RejectsReservedNames(string name)
    {
        Assert.False(InitCommand.IsValidModelName(name));
    }

    [Fact]
    public void IsValidModelName_RejectsOver50Chars()
    {
        var longName = new string('a', 51);
        Assert.False(InitCommand.IsValidModelName(longName));
    }

    [Fact]
    public void IsValidModelName_AcceptsMax50Chars()
    {
        var name = new string('a', 50);
        Assert.True(InitCommand.IsValidModelName(name));
    }

    // --- GetYamlTemplate tests ---

    [Theory]
    [InlineData("binary-classification", "accuracy")]
    [InlineData("multiclass-classification", "macro_accuracy")]
    [InlineData("regression", "r_squared")]
    public void GetYamlTemplate_ContainsCorrectMetric(string task, string expectedMetric)
    {
        var yaml = InitCommand.GetYamlTemplate("test-project", task, "default", "Label");

        Assert.Contains($"metric: {expectedMetric}", yaml);
    }

    [Fact]
    public void GetYamlTemplate_ContainsProjectName()
    {
        var yaml = InitCommand.GetYamlTemplate("my-project", "regression", "default", "Price");

        Assert.Contains("project: my-project", yaml);
        Assert.Contains("label: Price", yaml);
        Assert.Contains("default:", yaml);
    }

    [Fact]
    public void GetYamlTemplate_ContainsModelName()
    {
        var yaml = InitCommand.GetYamlTemplate("proj", "regression", "revenue", "Revenue");

        Assert.Contains("revenue:", yaml);
        Assert.Contains("label: Revenue", yaml);
    }
}
