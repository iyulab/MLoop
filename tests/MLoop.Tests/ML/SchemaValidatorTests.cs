using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class SchemaValidatorTests : IDisposable
{
    private readonly string _tempDir;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;

    public SchemaValidatorTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-sv-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private string CreateCsv(string name, string content)
    {
        var path = Path.Combine(_tempDir, name);
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }

    [Fact]
    public async Task ValidateAsync_EmptyInputFile_ReturnsResult()
    {
        var validator = new SchemaValidator(_fileSystem, _projectDiscovery);
        var inputPath = CreateCsv("empty.csv", "");

        var result = await validator.ValidateAsync(inputPath, "default");

        // Should handle gracefully without crashing
        Assert.NotNull(result);
    }

    [Fact]
    public void HeaderOf_AQuotedNameOverTwoLines_IsOneNameWithASpace()
    {
        var path = CreateCsv("multiline.csv", "번호,\"허용오차\n(상한값)\",\"A\r\nB\",결과\n1,0.5,2,통과\n");

        Assert.Equal(["번호", "허용오차 (상한값)", "A B", "결과"], SchemaValidator.HeaderOf(path)!);
    }

    private static MLoop.Core.Models.InputSchemaInfo Schema(params (string Name, string Purpose)[] columns) => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns = [.. columns.Select(c => new MLoop.Core.Models.ColumnSchema { Name = c.Name, Purpose = c.Purpose, DataType = "Numeric" })],
    };

    [Fact]
    public void AFeatureTheFileHasTwice_IsNamedWithItsPositions_AndAnExtraColumnMayBeRepeated()
    {
        var schema = Schema(("a", "Feature"), ("v", "Feature"), ("y", "Label"));

        var twice = SchemaValidator.FeatureColumnsNamedTwice(schema, ["a", "v", "v", "junk", "junk"]);

        var only = Assert.Single(twice);
        Assert.Equal("v", only.Name);
        Assert.Equal([2, 3], only.Positions);
        Assert.Empty(SchemaValidator.FeatureColumnsNamedTwice(schema, ["a", "v", "junk", "junk"]));
        // the label is not a model input in a prediction file: a repeated label column is not this finding
        Assert.Empty(SchemaValidator.FeatureColumnsNamedTwice(schema, ["a", "v", "y", "y"]));
    }

    [Fact]
    public void TheSameKoreanWordInTwoCompositions_IsTheSameColumn()
    {
        var composed = "한글";
        var schema = Schema((composed, "Feature"));

        var twice = SchemaValidator.FeatureColumnsNamedTwice(schema, [composed, composed.Normalize(System.Text.NormalizationForm.FormD)]);

        Assert.Equal([1, 2], Assert.Single(twice).Positions);
    }

    [Fact]
    public void HeaderOf_AnEmptyFile_IsNull()
    {
        Assert.Null(SchemaValidator.HeaderOf(CreateCsv("empty2.csv", "")));
    }

    [Fact]
    public async Task ValidateAsync_NoSavedSchema_SkipsAndSaysSo_RatherThanInventingAFinding()
    {
        // An experiment with no recorded input schema is not a data problem, and there is nothing to
        // compare the columns against. Reporting "missing columns" here would name a defect in data
        // that is fine — which is what comparing against the model artifact's own schema did: that
        // schema describes a featurized view, where the feature columns are folded into a single
        // vector and a label is present by construction, so present columns read as missing and
        // prediction data is asked for a label it is not supposed to carry.
        var validator = new SchemaValidator(_fileSystem, _projectDiscovery);
        var inputPath = CreateCsv("data.csv", "Feature1,Feature2,Label\n1,2,A\n3,4,B\n");

        var result = await validator.ValidateAsync(inputPath, "default");

        Assert.True(result.IsValid);
        Assert.Empty(result.MissingColumns);
        Assert.Contains("skipped", result.ErrorMessage, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(result.Suggestions);
    }

    [Fact]
    public void SchemaValidationResult_DefaultValues_AreCorrect()
    {
        var result = new SchemaValidationResult();

        Assert.False(result.IsValid); // default bool = false
        Assert.Null(result.ErrorMessage);
        Assert.NotNull(result.MissingColumns);
        Assert.Empty(result.MissingColumns);
        Assert.NotNull(result.Suggestions);
        Assert.Empty(result.Suggestions);
    }

    [Fact]
    public void UnicodeFormNote_says_which_spelling_each_side_has()
    {
        var composed = "지역";
        var decomposed = composed.Normalize(System.Text.NormalizationForm.FormD);

        var note = SchemaValidator.UnicodeFormNote([(composed, decomposed)]);

        Assert.Contains("Same text, different Unicode form", note);
        Assert.Contains($"Expected: '{composed}' (composed) → Found: '{decomposed}' (decomposed)", note);
    }
}
