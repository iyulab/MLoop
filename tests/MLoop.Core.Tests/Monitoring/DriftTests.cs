using System.Globalization;
using System.Text.Json;
using MLoop.Core.Models;
using MLoop.Core.Monitoring;

namespace MLoop.Core.Tests.Monitoring;

public class DriftProfileTests
{
    private static readonly (string, bool)[] Columns = [("x", true), ("kind", false)];

    private static string?[][] Rows(int n, Func<int, string?> x, Func<int, string?> kind)
        => Enumerable.Range(0, n).Select(i => new[] { x(i), kind(i) }).ToArray();

    [Fact]
    public void ANumericColumn_IsCutAtItsDeciles()
    {
        var profile = DriftProfile.Build(Rows(1000, i => (i + 1).ToString(CultureInfo.InvariantCulture), _ => "a"), Columns);

        var x = profile.Features[0];
        Assert.True(x.IsNumeric);
        Assert.Equal(9, x.Edges!.Count);
        Assert.Equal(11, x.BinCount);                       // ten value bins + missing
        Assert.All(x.Shares.Take(10), s => Assert.Equal(0.1, s, 2));
        Assert.Equal(0, x.Shares[^1]);
    }

    [Fact]
    public void RepeatedValues_DoNotSplitAcrossBins()
    {
        // 90% zeros: every decile edge below the 90th is 0, and appears once.
        var profile = DriftProfile.Build(Rows(1000, i => i < 900 ? "0" : i.ToString(CultureInfo.InvariantCulture), _ => "a"), Columns);

        var x = profile.Features[0];
        Assert.Equal(x.Edges!.Distinct().Count(), x.Edges!.Count);
        Assert.Equal(0.9, x.Shares[x.BinOf("0")], 3);
    }

    // The model reads every number as a single-precision float: the same measurement written 5 and 4.999999999999999 is one value, and a
    // training file's 0.999999993939 is the 1 that the prediction log records. Compared in double precision, rows drawn from the training
    // data would sit in another bin than the one the profile counted them in, and read as drift.
    [Fact]
    public void ValuesThatDifferBelowSinglePrecision_FallInOneBin()
    {
        var profile = DriftProfile.Build(Rows(1000, i => i % 10 == 4 ? (i % 20 == 4 ? "4.999999999999999" : "5") : (i % 10).ToString(CultureInfo.InvariantCulture), _ => "a"), Columns);
        var x = profile.Features[0];
        Assert.Equal(x.BinOf("5"), x.BinOf("4.999999999999999"));

        // The prediction log writes the single-precision value the model saw; the training rows, logged, must fill the bins the profile counted them in.
        var written = Enumerable.Range(0, 1000).Select(i => (1 - 1e-9 * (1 + i % 50)).ToString("R", CultureInfo.InvariantCulture)).ToList();
        var ratio = DriftProfile.Build(written.Select(v => new[] { v, "a" }).ToArray(), Columns).Features[0];
        var logged = written.Select(v => ((float)double.Parse(v, CultureInfo.InvariantCulture)).ToString("R", CultureInfo.InvariantCulture)).ToList();
        Assert.Equal(ratio.BinOf("1"), ratio.BinOf("0.999999993939"));
        Assert.True(DriftMeasure.Psi(ratio.Shares, ratio.SharesOf(logged)) < 1e-9);
    }

    [Fact]
    public void ACategoricalColumn_KeepsItsCommonCategories_AndCountsTheRestAsOther()
    {
        var profile = DriftProfile.Build(Rows(1000, _ => "1", i => i % 50 == 0 ? null : $"k{i % 25}"), Columns);

        var kind = profile.Features[1];
        Assert.False(kind.IsNumeric);
        Assert.Equal(FeatureProfile.MaxCategories, kind.Categories!.Count);
        Assert.Equal(FeatureProfile.MaxCategories, kind.BinOf("never-seen"));     // other
        Assert.Equal(kind.BinCount - 1, kind.BinOf(""));                           // missing
        Assert.Equal(0.02, kind.Shares[^1], 3);
        Assert.Equal(kind.BinOf("K3"), kind.BinOf("k3"));                          // case does not split a value
        Assert.Equal(1.0, kind.Shares.Sum(), 9);
    }

    // A yes/no column's training CSV says "yes"/"no"; a JSON request sends true/false and is logged as
    // True/False. The loader reads both as one boolean, so the profile must bin them together.
    [Theory]
    [InlineData("True", "yes")]
    [InlineData("false", "no")]
    [InlineData("Y", "yes")]
    [InlineData("0", "no")]
    public void A_yes_no_value_bins_with_its_meaning_whatever_its_spelling(string served, string trained)
    {
        var flag = FeatureProfile.Categorical("flag", ["yes", "no", "yes", "no", "yes"]);

        Assert.Equal(flag.BinOf(trained), flag.BinOf(served));
    }

    // Only in a yes/no column: one category that happens to read as yes/no does not absorb every true.
    [Fact]
    public void A_category_that_merely_looks_yes_no_keeps_its_own_values()
    {
        var plan = FeatureProfile.Categorical("plan", ["Yes", "Basic", "Pro"]);

        Assert.Equal(plan.Categories!.Count, plan.BinOf("true")); // other
    }

    [Fact]
    public void ColumnsOf_TakesNumericCategoricalAndBooleanFeatures_NotTextLabelOrExcluded()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns =
            [
                new ColumnSchema { Name = "n", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
                new ColumnSchema { Name = "c", DataType = SchemaDataTypes.Categorical, Purpose = "Feature" },
                new ColumnSchema { Name = "b", DataType = SchemaDataTypes.Boolean, Purpose = "Feature" },
                new ColumnSchema { Name = "t", DataType = SchemaDataTypes.Text, Purpose = "Feature" },
                new ColumnSchema { Name = "y", DataType = SchemaDataTypes.Categorical, Purpose = "Label" },
                new ColumnSchema { Name = "id", DataType = SchemaDataTypes.Numeric, Purpose = "Exclude" },
            ],
        };

        Assert.Equal([("n", true), ("c", false), ("b", false)], DriftProfile.ColumnsOf(schema));
    }

    [Fact]
    public async Task FromCsv_SamplesALargeFile_TheSameWayEveryTime()
    {
        var path = Path.Combine(Path.GetTempPath(), $"drift-{Guid.NewGuid():N}.csv");
        try
        {
            await File.WriteAllLinesAsync(path,
                new[] { "X,Kind,label" }.Concat(Enumerable.Range(0, 5000).Select(i => $"{i},{(i % 2 == 0 ? "even" : "odd")},1")));

            var first = await DriftProfile.FromCsvAsync(path, Columns, sampleSize: 1000);
            var second = await DriftProfile.FromCsvAsync(path, Columns, sampleSize: 1000);

            Assert.Equal(1000, first.Rows);
            Assert.Equal(first.Features[0].Edges, second.Features[0].Edges);
            Assert.InRange(first.Features[0].Edges![4], 2000, 3000);                // a sample of the whole file, not its head
            Assert.Equal(0.5, first.Features[1].Shares[first.Features[1].BinOf("even")], 1);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

public class DriftMeasureTests
{
    private static readonly (string, bool)[] Columns = [("x", true), ("kind", false)];

    private static DriftProfile Reference()
        => DriftProfile.Build(Enumerable.Range(0, 2000)
            .Select(i => new[] { (i % 100).ToString(CultureInfo.InvariantCulture), i % 2 == 0 ? "a" : "b" }).ToArray(), Columns);

    private static IReadOnlyDictionary<string, object?> Row(object? x, object? kind)
        => new Dictionary<string, object?> { ["x"] = x, ["kind"] = kind };

    [Theory]
    [InlineData(0.05, DriftLevel.None)]
    [InlineData(0.1, DriftLevel.Moderate)]
    [InlineData(0.25, DriftLevel.Significant)]
    public void Levels_FollowTheUsualReading(double psi, DriftLevel level)
        => Assert.Equal(level, DriftMeasure.LevelOf(psi));

    [Fact]
    public void Psi_IsZeroForTheSameDistribution_AndSymmetric()
    {
        double[] p = [0.2, 0.3, 0.5], q = [0.4, 0.4, 0.2];

        Assert.Equal(0, DriftMeasure.Psi(p, p), 12);
        Assert.Equal(DriftMeasure.Psi(p, q), DriftMeasure.Psi(q, p), 12);
        Assert.True(DriftMeasure.Psi(p, q) > 0);
    }

    [Fact]
    public void TheSameInputs_DoNotDrift()
    {
        var rows = Enumerable.Range(0, 2000).Select(i => Row(i % 100, i % 2 == 0 ? "a" : "b")).ToList();

        var report = DriftMeasure.Compare(Reference(), rows);

        Assert.Equal(DriftLevel.None, report.Level);
        Assert.True(report.MaxPsi < 0.01, $"PSI {report.MaxPsi}");
    }

    // A column capped at its maximum (half of the training rows sit on it): the end bin is open, so a window that left the range altogether
    // used to land in the same bin as the capped rows and move its share by half. Counted apart, it is the plain drift it is.
    [Fact]
    public void AWindowPastACappedMaximum_Drifts_AndNamesTheRangeItLeft()
    {
        var trained = Enumerable.Range(0, 2000).Select(i => new[] { (i % 2 == 0 ? 8 : i % 8).ToString(CultureInfo.InvariantCulture), "a" }).ToArray();
        var profile = DriftProfile.Build(trained, Columns);
        var inside = Enumerable.Range(0, 600).Select(i => Row(i % 2 == 0 ? 8 : i % 8, "a")).ToList();
        // a third of the window left the range, the rest is as trained
        var past = inside.Take(400).Concat(Enumerable.Range(0, 200).Select(i => Row(9 + i % 7, "a"))).ToList();

        var report = DriftMeasure.Compare(profile, past);
        Assert.Equal(DriftLevel.Significant, report.Level);
        Assert.Equal("> 8", report.Features.First(f => f.Column == "x").MostChangedBin);
        Assert.Equal(DriftLevel.None, DriftMeasure.Compare(profile, inside).Level);

        // a profile written before the range was kept reads as it did: the window shares the capped rows' bin and the shift stays small
        var legacy = profile with { Features = [.. profile.Features.Select(f => f with { Min = null, Max = null })] };
        Assert.True(DriftMeasure.Compare(legacy, past).Features.First(f => f.Column == "x").Psi < DriftMeasure.SignificantThreshold);
    }

    [Fact]
    public void ANewValueInAConstantColumn_Drifts()
    {
        var profile = DriftProfile.Build(Enumerable.Range(0, 500).Select(_ => new[] { "5", "a" }).ToArray(), Columns);

        Assert.Equal(DriftLevel.None, DriftMeasure.Compare(profile, Enumerable.Range(0, 400).Select(_ => Row(5, "a")).ToList()).Level);
        Assert.Equal(DriftLevel.Significant, DriftMeasure.Compare(profile, Enumerable.Range(0, 400).Select(_ => Row(6, "a")).ToList()).Level);
    }

    [Fact]
    public void ShiftedNumbersAndANewCategory_Drift_AndSayWhere()
    {
        // x moves to its top half; a third of the kinds is a category training never saw.
        var rows = Enumerable.Range(0, 900).Select(i => Row(50 + i % 50, i % 3 == 0 ? "c" : "a")).ToList();

        var report = DriftMeasure.Compare(Reference(), rows);

        Assert.Equal(DriftLevel.Significant, report.Level);
        var kind = report.Features.Single(f => f.Column == "kind");
        Assert.Equal(DriftLevel.Significant, kind.Level);
        Assert.Equal("b", kind.MostChangedBin);        // b vanished: 0.5 → 0
        Assert.Equal(0.5, kind.ReferenceShare, 2);
        Assert.Equal(0, kind.CurrentShare);
        Assert.Equal(DriftLevel.Significant, report.Features.Single(f => f.Column == "x").Level);
        Assert.Equal(report.Features.Max(f => f.Psi), report.Features[0].Psi);   // most moved first
    }

    [Theory]
    [InlineData("binary-classification", false)]
    [InlineData("multiclass-classification", false)]
    [InlineData("regression", true)]
    public void AnAnswerProfile_IsCategoricalForAClassifier_AndNumericForARegressor(string task, bool numeric)
    {
        var answers = Enumerable.Range(0, 100).Select(i => (object?)(numeric ? i * 1.5 : i % 4 == 0 ? "yes" : "no")).ToList();

        var profile = DriftProfile.PredictionProfileOf(task, answers);

        Assert.NotNull(profile);
        Assert.Equal(numeric, profile!.IsNumeric);
        Assert.Equal(DriftProfile.PredictionColumn, profile.Column);
        Assert.Equal(1.0, profile.Shares.Sum(), 9);
    }

    [Theory]
    [InlineData("clustering")]
    [InlineData("anomaly-detection")]
    [InlineData("ranking")]
    public void ATaskWhoseAnswerIsNotAClassOrAValue_HasNoAnswerProfile(string task)
        => Assert.Null(DriftProfile.PredictionProfileOf(task, [1.0, 2.0]));

    [Fact]
    public void ShiftedAnswers_Drift_ButDoNotRaiseTheInputLevel()
    {
        var answersAtTraining = Enumerable.Range(0, 2000).Select(i => (object?)(i % 4 == 0 ? "yes" : "no")).ToList();
        var profile = Reference() with { Prediction = DriftProfile.PredictionProfileOf("binary-classification", answersAtTraining) };
        var rows = Enumerable.Range(0, 2000).Select(i => Row(i % 100, i % 2 == 0 ? "a" : "b")).ToList();
        // The same inputs, now answered "yes" twice as often — and logged as a JSON string, case changed.
        using var doc = JsonDocument.Parse("""["YES", "no"]""");
        var yes = doc.RootElement[0].Clone();
        var answers = Enumerable.Range(0, 2000).Select(i => (object?)(i % 2 == 0 ? yes : "no")).ToList();

        var report = DriftMeasure.Compare(profile, rows, answers);

        Assert.Equal(DriftLevel.None, report.Level);
        Assert.NotNull(report.Prediction);
        Assert.Equal(DriftLevel.Significant, report.Prediction!.Level);
        Assert.Equal(DriftProfile.PredictionColumn, report.Prediction.Column);
        // Two answers move by the same amount; the tie goes to the more common one, "no": 0.75 → 0.5.
        Assert.Equal("no", report.Prediction.MostChangedBin);
        Assert.Equal(0.75, report.Prediction.ReferenceShare, 2);
        Assert.Equal(0.5, report.Prediction.CurrentShare, 2);
    }

    [Fact]
    public void AProfileWrittenBeforeAnswersWereProfiled_ComparesItsInputsOnly()
    {
        var json = JsonSerializer.Serialize(Reference(), new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase })
            .Replace(",\"prediction\":null", "");
        var old = JsonSerializer.Deserialize<DriftProfile>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        var rows = Enumerable.Range(0, 200).Select(i => Row(i % 100, "a")).ToList();

        var report = DriftMeasure.Compare(old, rows, rows.Select(_ => (object?)"yes").ToList());

        Assert.DoesNotContain("prediction", json, StringComparison.OrdinalIgnoreCase);
        Assert.Null(old.Prediction);
        Assert.Null(report.Prediction);
        Assert.NotEmpty(report.Features);
    }

    [Fact]
    public void LoggedJsonValues_AndAbsentColumns_AreReadLikeTheTrainingFile()
    {
        using var doc = JsonDocument.Parse("""[{"x": 7, "kind": "a"}, {"x": "7", "kind": "A"}]""");
        var json = doc.RootElement.EnumerateArray()
            .Select(e => (IReadOnlyDictionary<string, object?>)e.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value.Clone()))
            .ToList();
        var profile = Reference();

        var report = DriftMeasure.Compare(profile, json);
        var absent = DriftMeasure.Compare(profile, [new Dictionary<string, object?> { ["X"] = 7 }]);

        // 7 and "7", "a" and "A" are one value each: every row lands in one x bin and in "a".
        var kind = report.Features.Single(f => f.Column == "kind");
        Assert.Equal("a", kind.MostChangedBin);              // a and b move by 0.5 each; the first wins the tie
        Assert.Equal(1.0, kind.CurrentShare);
        Assert.Equal(1.0, report.Features.Single(f => f.Column == "x").CurrentShare);
        // A column the row does not carry is missing; one it carries under another case is read.
        Assert.Equal("(missing)", absent.Features.Single(f => f.Column == "kind").MostChangedBin);
        Assert.Equal(1.0, absent.Features.Single(f => f.Column == "x").CurrentShare);
    }
}

public class TextProfileTests
{
    private static readonly string[] Topics = ["delivery was late again", "the screen cracked on day one", "refund took three weeks",
        "battery drains overnight", "support answered quickly", "wrong size was shipped", "great value for the price"];

    private static List<string?> Reviews(int n, string[] pool, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (string?)$"{pool[random.Next(pool.Length)]} {pool[random.Next(pool.Length)]}").ToList();
    }

    private static IReadOnlyDictionary<string, object?> Row(string? text) => new Dictionary<string, object?> { ["review"] = text };

    private static DriftProfile Profile(IReadOnlyList<string?> values)
        => new() { Rows = values.Count, Features = [], Texts = [TextProfile.Build("review", values)] };

    [Fact]
    public void Words_AreLowerCasedLetterRuns_AndUnspacedScriptsGoCharacterByCharacter()
    {
        Assert.Equal(["delivery", "was", "late", "2", "days"], TextProfile.Words("Delivery was LATE: 2 days!"));
        Assert.Equal(["배송이", "늦었어요"], TextProfile.Words("배송이 늦었어요."));
        Assert.Equal(["配", "送", "abc"], TextProfile.Words("配送abc"));
        Assert.Empty(TextProfile.Words("  ...  "));
    }

    [Fact]
    public void TheSameKindOfText_DoesNotDrift()
    {
        var profile = Profile(Reviews(2000, Topics, 1));

        var report = DriftMeasure.Compare(profile, Reviews(1000, Topics, 2).Select(Row).ToList());

        Assert.Equal([TextProfile.LengthColumn("review"), TextProfile.UnseenColumn("review")],
            report.Features.Select(f => f.Column).Order(StringComparer.Ordinal));
        Assert.Equal(DriftLevel.None, report.Level);
    }

    [Fact]
    public void TextAboutWhatTheModelNeverRead_DriftsInItsUnseenWords()
    {
        var profile = Profile(Reviews(2000, Topics, 1));
        string[] elsewhere = ["firmware bricked the router", "invoice currency mismatch", "warehouse inventory audit pending"];

        var report = DriftMeasure.Compare(profile, Reviews(1000, elsewhere, 3).Select(Row).ToList());

        var unseen = report.Features.Single(f => f.Column == TextProfile.UnseenColumn("review"));
        Assert.Equal(DriftLevel.Significant, unseen.Level);
        Assert.Equal(DriftLevel.Significant, report.Level);   // a text column moves input_drift like any column
    }

    [Fact]
    public void TheUnseenReference_IsNotZero_WhenEveryWordIsRare()
    {
        // Every value distinct: measured on the rows its vocabulary came from, every row would read 0 % unseen.
        var values = Enumerable.Range(0, 400).Select(i => (string?)$"token{i} other{i}").ToList();

        var profile = TextProfile.Build("review", values);

        Assert.True(profile.Unseen.Shares[profile.Unseen.BinOf("1")] > 0.9);
    }

    [Fact]
    public void Of_ProfilesTextFeaturesApart_FromBinnedColumns()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns =
            [
                new ColumnSchema { Name = "n", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
                new ColumnSchema { Name = "review", DataType = SchemaDataTypes.Text, Purpose = "Feature" },
                new ColumnSchema { Name = "y", DataType = SchemaDataTypes.Text, Purpose = "Label" },
            ],
        };
        var sample = new CsvSample(["n", "review", "y"], Enumerable.Range(0, 50).Select(i => new string?[] { i.ToString(CultureInfo.InvariantCulture), $"word{i % 4} more", "a" }).ToList());

        var profile = DriftProfile.Of(sample, schema);

        Assert.Equal(["n"], profile.Features.Select(f => f.Column));
        Assert.Equal(["review"], profile.Texts!.Select(t => t.Column));
        Assert.Equal(["more", "word0", "word2"], profile.Texts![0].Vocabulary.Order(StringComparer.Ordinal));   // even rows only
    }

    [Fact]
    public void AProfileWithText_RoundTripsThroughJson()
    {
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true };
        var profile = Profile(Reviews(200, Topics, 1));

        var back = JsonSerializer.Deserialize<DriftProfile>(JsonSerializer.Serialize(profile, options), options)!;

        Assert.Equal(profile.Texts![0].Vocabulary, back.Texts![0].Vocabulary);
        Assert.Equal(profile.Texts[0].Unseen.Shares, back.Texts[0].Unseen.Shares);
    }
}
