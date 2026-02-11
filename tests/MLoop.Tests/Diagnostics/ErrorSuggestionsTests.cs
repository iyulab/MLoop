using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Diagnostics;

public class ErrorSuggestionsTests
{
    [Fact]
    public void GetSuggestions_FileNotFound_SuggestsCheckPath()
    {
        var ex = new FileNotFoundException("Could not find file 'test.csv'");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.NotEmpty(suggestions);
        Assert.Contains(suggestions, s => s.Contains("file path"));
    }

    [Fact]
    public void GetSuggestions_FileNotFoundInTraining_SuggestsDataFile()
    {
        var ex = new FileNotFoundException("File not found");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.Contains(suggestions, s => s.Contains("datasets/train.csv"));
    }

    [Fact]
    public void GetSuggestions_SchemaError_SuggestsColumnCheck()
    {
        var ex = new Exception("Schema mismatch: column 'price' not found");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.Contains(suggestions, s => s.Contains("columns"));
    }

    [Fact]
    public void GetSuggestions_LabelNotFound_SuggestsLabelUpdate()
    {
        var ex = new Exception("Label column not found in data");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.Contains(suggestions, s => s.Contains("label"));
    }

    [Fact]
    public void GetSuggestions_PromoteContext_SuggestsList()
    {
        var ex = new FileNotFoundException("Experiment not found: default/exp-999");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "promote");

        Assert.Contains(suggestions, s => s.Contains("mloop list"));
    }

    [Fact]
    public void GetSuggestions_PreprocessingContext_SuggestsDryRun()
    {
        var ex = new InvalidOperationException("Unknown prep step type");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "preprocessing");

        Assert.Contains(suggestions, s => s.Contains("dry-run"));
    }

    [Fact]
    public void GetSuggestions_ConfigError_SuggestsYamlCheck()
    {
        var ex = new Exception("Invalid mloop.yaml configuration");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.Contains(suggestions, s => s.Contains("yaml") || s.Contains("YAML"));
    }

    [Fact]
    public void GetSuggestions_UnknownError_ReturnsFallback()
    {
        var ex = new Exception("Something completely unexpected happened");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.NotEmpty(suggestions);
        Assert.Contains(suggestions, s => s.Contains("github.com") || s.Contains("Review"));
    }
}
