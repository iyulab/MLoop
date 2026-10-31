using System.Text;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Hangul written as syllables (NFC) or as jamo (NFD) is the same text and different bytes. A file saved on macOS arrives
/// decomposed; compared as bytes, a category <c>서울</c> was a value training never saw (replaced by the most frequent one, or —
/// in <c>evaluate</c>, which scores the file as it is — silently scored as unknown: R² 0.9997 became 0.48) and a column
/// <c>지역</c> was "missing" from a file that visibly has it.
/// </summary>
public class CanonicalTextTests : IDisposable
{
    private static string Nfd(string text) => text.Normalize(NormalizationForm.FormD);

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop-canonical-{Guid.NewGuid():N}");

    public CanonicalTextTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, true);
    }

    [Fact]
    public void The_two_forms_are_different_strings_and_the_same_text()
    {
        var composed = "서울";
        var decomposed = Nfd(composed);

        Assert.NotEqual(composed, decomposed);          // the premise: the bytes differ
        Assert.True(CanonicalText.SameText(composed, decomposed));
        Assert.False(CanonicalText.SameText(composed, composed));   // identical is not "another form"
        Assert.False(CanonicalText.SameText("서울", "부산"));
        Assert.Equal(composed, CanonicalText.Nfc(decomposed));
        Assert.Same(composed, CanonicalText.Nfc(composed));          // nothing to do, nothing allocated
    }

    [Fact]
    public void A_value_resolves_to_the_spelling_training_saw()
    {
        var trained = new[] { "서울", "부산", "대구" };
        var exact = new HashSet<string>(trained);
        var byNfc = CanonicalText.LookupByNfc(trained);

        Assert.Equal("서울", CanonicalText.Resolve(exact, byNfc, "서울"));
        Assert.Equal("서울", CanonicalText.Resolve(exact, byNfc, Nfd("서울")));
        Assert.Null(CanonicalText.Resolve(exact, byNfc, "광주"));       // a value training never saw stays unknown
        Assert.Null(CanonicalText.Resolve(exact, byNfc, Nfd("광주")));
    }

    [Fact]
    public void Training_on_decomposed_text_is_answered_in_the_trained_spelling()
    {
        var trained = new[] { Nfd("서울"), Nfd("부산") };
        var exact = new HashSet<string>(trained);
        var byNfc = CanonicalText.LookupByNfc(trained);

        Assert.Equal(Nfd("서울"), CanonicalText.Resolve(exact, byNfc, "서울"));   // a composed file against an older, decomposed model
    }

    private static InputSchemaInfo Schema() => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "지역", DataType = "Categorical", Purpose = "Feature", CategoricalValues = ["대구", "부산", "서울"], UniqueValueCount = 3 },
            new ColumnSchema { Name = "면적", DataType = "Numeric", Purpose = "Feature" },
        ]
    };

    [Fact]
    public void Served_rows_with_decomposed_categories_are_the_trained_value_with_no_warning()
    {
        var rows = new[]
        {
            new Dictionary<string, object> { ["지역"] = Nfd("서울"), ["면적"] = 1.0f },
            new Dictionary<string, object> { ["지역"] = "광주", ["면적"] = 2.0f },     // a real unknown
        };
        var warnings = new List<string>();

        PredictionService.MapCategoricalValues(rows, Schema(), warnings);

        Assert.Equal("서울", rows[0]["지역"]);
        Assert.Equal("대구", rows[1]["지역"]);                // still replaced by the fallback...
        var only = Assert.Single(warnings);                    // ...and still said
        Assert.Contains("광주", only);
    }

    [Fact]
    public void A_prediction_file_with_decomposed_categories_has_no_unknown_values()
    {
        var file = Path.Combine(_dir, "predict.csv");
        File.WriteAllText(file, $"지역,면적\n{Nfd("서울")},1\n{Nfd("부산")},2\n", new UTF8Encoding(false));

        var result = new CategoricalMapper().PreprocessPredictionData(file, Schema(), CategoricalMapper.UnknownValueStrategy.Error);

        Assert.True(result.Success, result.ErrorMessage);
        Assert.Empty(result.UnknownValues);
        var written = File.ReadAllText(result.TempFilePath!);
        // Ordinal: the culture-aware comparison xunit's Contains defaults to treats the two forms as equal.
        Assert.True(written.Contains("서울", StringComparison.Ordinal));
        Assert.False(written.Contains(Nfd("서울"), StringComparison.Ordinal));
    }
}
