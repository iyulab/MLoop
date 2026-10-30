using System.Globalization;
using System.Text.Json;
using FluentAssertions;
using MLoop.Core.Monitoring;
using MLoop.Core.Storage;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class DriftEvaluatorTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-drift-{Guid.NewGuid():N}");
    private readonly FilePredictionLogger _logs;

    public DriftEvaluatorTests()
    {
        Directory.CreateDirectory(_root);
        _logs = new FilePredictionLogger(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private string Production => Path.Combine(_root, ExperimentLayout.ModelsDirectory, "m", ExperimentLayout.ProductionDirectory);

    private void Promote(string experimentId, bool withProfile = true, double? yesShareAtTraining = null, int profileRows = 1000)
    {
        Directory.CreateDirectory(Production);
        File.WriteAllText(Path.Combine(Production, ExperimentLayout.MetadataFileName),
            JsonSerializer.Serialize(new { experimentId }, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
        if (!withProfile)
            return;
        var rows = Enumerable.Range(0, profileRows).Select(i => new[] { (i % 100).ToString(CultureInfo.InvariantCulture) }).ToArray();
        var profile = DriftProfile.Build(rows, [("x", true)]);
        if (yesShareAtTraining is { } share)
            profile = profile with
            {
                Prediction = DriftProfile.PredictionProfileOf("binary-classification",
                    Enumerable.Range(0, 1000).Select(i => (object?)(i < share * 1000 ? "yes" : "no")).ToList()),
            };
        File.WriteAllText(Path.Combine(Production, ExperimentLayout.DriftProfileFileName),
            JsonSerializer.Serialize(profile, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
    }

    private async Task Log(string experimentId, int count, Func<int, double> x, Func<int, string>? site = null, Func<int, string>? answer = null)
    {
        for (var i = 0; i < count; i++)
        {
            var input = new Dictionary<string, object> { ["x"] = x(i) };
            if (site is not null)
                input["site"] = site(i);
            await _logs.LogPredictionAsync("m", experimentId, input, answer?.Invoke(i) ?? "1");
        }
    }

    [Theory]
    [InlineData(100, true)]
    [InlineData(150, false)]
    [InlineData(1000, false)]
    public async Task ASmallTrainingProfile_IsReadButTheAnswerSaysHowMuchItWeighs(int profileRows, bool cautions)
    {
        Promote("exp-001", profileRows: profileRows);
        await Log("exp-001", DriftEvaluator.MinimumRows, i => i % 100);

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m");

        check.Status.Should().Be(DriftCheckStatus.Measured);
        (check.Message?.Contains($"training profile holds only {profileRows} rows") ?? false).Should().Be(cautions);
    }

    [Fact]
    public async Task WithoutProduction_ItSaysSo()
        => (await DriftEvaluator.CheckAsync(_root, _logs, "m")).Status.Should().Be(DriftCheckStatus.NoProduction);

    [Fact]
    public async Task AModelWithoutAProfile_IsNotMeasured_AndTheMessageSaysHowToStart()
    {
        Promote("exp-001", withProfile: false);

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m");

        check.Status.Should().Be(DriftCheckStatus.NoProfile);
        check.Message.Should().Contain("Retrain and promote");
    }

    [Fact]
    public async Task TooFewPredictions_AreNotCompared()
    {
        Promote("exp-001");
        await Log("exp-001", 50, i => i);

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m");

        check.Status.Should().Be(DriftCheckStatus.TooFewRows);
        check.Rows.Should().Be(50);
    }

    [Fact]
    public async Task OnlyTheProductionModelsOwnPredictions_AreCompared()
    {
        Promote("exp-002");
        await Log("exp-001", 300, _ => 1000);      // an earlier model's traffic, far off
        await Log("exp-002", 300, i => i % 100);

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m");

        check.Status.Should().Be(DriftCheckStatus.Measured);
        check.Rows.Should().Be(300);
        check.Report!.Level.Should().Be(DriftLevel.None);
    }

    [Fact]
    public async Task ASlice_ShowsWhereTheInputsMoved()
    {
        Promote("exp-001");
        await Log("exp-001", 800, i => i % 2 == 0 ? i % 100 : 95 + i % 5, i => i % 2 == 0 ? "steady" : "moved");

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m", sliceColumn: "site");

        check.Slices!.Select(s => s.Value).Should().Equal("moved", "steady");
        check.Slices![0].Report.Level.Should().Be(DriftLevel.Significant);
        check.Slices![1].Report.Level.Should().Be(DriftLevel.None);
    }

    [Fact]
    public async Task TheModelsAnswers_AreComparedWithItsAnswersAtTraining()
    {
        Promote("exp-001", yesShareAtTraining: 0.2);
        await Log("exp-001", 400, i => i % 100, answer: i => i % 2 == 0 ? "yes" : "no");

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m");

        check.Report!.Level.Should().Be(DriftLevel.None);            // the inputs held
        check.Report.Prediction!.Level.Should().Be(DriftLevel.Significant);
        check.Report.Prediction.MostChangedBin.Should().Be("no");   // 0.8 → 0.5 (yes moves as much the other way)
        check.Message.Should().Contain("Answers: significant drift");
        var signal = JsonSerializer.SerializeToElement(DriftEvaluator.Signal(check.Report));
        signal.GetProperty("prediction").GetProperty("column").GetString().Should().Be(DriftProfile.PredictionColumn);
        signal.GetProperty("level").GetString().Should().Be("None");
    }

    [Fact]
    public async Task WithoutAnAnswerProfile_TheSignalSaysSo()
    {
        Promote("exp-001");
        await Log("exp-001", 300, i => i % 100);

        var check = await DriftEvaluator.CheckAsync(_root, _logs, "m");

        check.Report!.Prediction.Should().BeNull();
        JsonSerializer.SerializeToElement(DriftEvaluator.Signal(check.Report)).GetProperty("prediction").ValueKind
            .Should().Be(JsonValueKind.Null);
    }

    [Fact]
    public async Task TheTrigger_MeetsTheDriftCondition_OnSignificantDrift()
    {
        Promote("exp-001");
        await Log("exp-001", 300, _ => 99);
        var trigger = new CompositeRetrainingTrigger(_root, new FileFeedbackCollector(_root), _logs);

        var result = (await trigger.EvaluateAsync("m", [RetrainingDefaults.Drift()])).ConditionResults.Single();

        result.IsMet.Should().BeTrue();
        result.Drift!.Features.Single().Column.Should().Be("x");
        result.CurrentValue.Should().BeGreaterThanOrEqualTo(RetrainingDefaults.DriftThreshold);
    }
}
