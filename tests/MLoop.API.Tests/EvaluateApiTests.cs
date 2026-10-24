using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /evaluate</c> weighs the rows the way <c>mloop evaluate</c> does and compares two models on
/// the same rows the same way — the endpoint once evaluated a weighted experiment unweighted, with
/// nothing in the response saying so. Its own fixture: the experiments it trains would otherwise show
/// up in the suites that assert an empty project.
/// </summary>
public class EvaluateApiTests : IClassFixture<TestWebApplicationFactory>, IAsyncLifetime
{
    private const string Model = "weighted";
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;
    private string _withWeight = "";
    private string _withoutWeight = "";

    public EvaluateApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    public async ValueTask InitializeAsync()
    {
        var datasets = Path.Combine(_factory.TestProjectRoot, "datasets");
        _withWeight = Path.Combine(datasets, "weighted.csv");
        _withoutWeight = Path.Combine(datasets, "unweighted.csv");
        if (File.Exists(_withWeight))
            return; // the class fixture is shared by every test in this class; train once

        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(3);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        await File.WriteAllTextAsync(_withWeight, csv.ToString());
        await File.WriteAllLinesAsync(_withoutWeight,
            (await File.ReadAllLinesAsync(_withWeight)).Select(l => string.Join(',', l.Split(',').Where((_, i) => i != 2))));

        var fs = _factory.Services.GetRequiredService<IFileSystemManager>();
        var store = _factory.Services.GetRequiredService<IExperimentStore>();
        // Training narrates to the console; this assembly runs in parallel, so the console is left alone.
        for (var run = 0; run < 2; run++)
            await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = Model,
                DataFile = _withWeight,
                LabelColumn = "y",
                Task = "binary-classification",
                WeightColumn = "w",
                TimeLimitSeconds = 5,
            }, progress: null, CancellationToken.None);
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    private async Task<(HttpStatusCode Status, JsonElement Body)> EvaluateAsync(object request)
    {
        var response = await _client.PostAsJsonAsync("/evaluate", request);
        var text = await response.Content.ReadAsStringAsync();
        return (response.StatusCode, JsonDocument.Parse(text).RootElement.Clone());
    }

    /// <summary>
    /// An experiment that kept its record but not a model is refused — it used to be evaluated with
    /// production's model while the report carried the experiment's id.
    /// </summary>
    [Fact]
    public async Task AnExperimentWithoutAModel_IsRefused_NotScoredWithProductionsModel()
    {
        var staging = Path.Combine(_factory.TestProjectRoot, "models", Model, "staging");
        var withoutModel = Path.Combine(staging, "exp-090");
        Directory.CreateDirectory(withoutModel);
        foreach (var file in new[] { "metadata.json", "config.json" })
            if (File.Exists(Path.Combine(staging, "exp-001", file)))
                File.WriteAllText(Path.Combine(withoutModel, file),
                    File.ReadAllText(Path.Combine(staging, "exp-001", file)).Replace("exp-001", "exp-090"));

        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-090", testDataPath = _withWeight });

        status.Should().Be(HttpStatusCode.NotFound, body.ToString());
        body.GetProperty("error").GetString().Should().Contain("no trained model");
    }

    [Fact]
    public async Task AWeightedExperiment_IsScoredByItsWeight_WhenTheFileHasIt()
    {
        var holdout = Path.Combine(_factory.TestProjectRoot, "datasets", "weighted-holdout.csv");
        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(11);
        for (var i = 0; i < 80; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        await File.WriteAllTextAsync(holdout, csv.ToString());

        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = holdout });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        body.GetProperty("weightColumn").GetString().Should().Be("w");
        body.TryGetProperty("warnings", out _).Should().BeFalse("the API leaves null fields out");
    }

    /// <summary>
    /// The file the experiment was trained on — the same content, under any name — is mostly training
    /// rows: the answer says so before any other warning.
    /// </summary>
    [Fact]
    public async Task TheFileTheModelWasTrainedOn_IsScored_WithAWarningThatItIsTrainingData()
    {
        var renamed = Path.Combine(_factory.TestProjectRoot, "datasets", "same-content.csv");
        File.Copy(_withWeight, renamed, overwrite: true);

        foreach (var path in new[] { _withWeight, renamed })
        {
            var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = path });

            status.Should().Be(HttpStatusCode.OK, body.ToString());
            body.GetProperty("warnings")[0].GetString().Should().Contain("the file the model was trained on");
        }
    }

    [Fact]
    public async Task SliceBy_MeasuresEachValue_AsMloopEvaluateSliceByDoes()
    {
        // The test rows tagged by site: the endpoint reports each site's metrics beside the overall ones.
        var lines = await File.ReadAllLinesAsync(_withWeight);
        var sited = Path.Combine(_factory.TestProjectRoot, "datasets", "sited.csv");
        await File.WriteAllLinesAsync(sited, lines.Select((l, i) => l + "," + (i == 0 ? "site" : i % 2 == 0 ? "east" : "west")));

        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = sited, sliceBy = "site" });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        body.GetProperty("sliceBy").GetString().Should().Be("site");
        var slices = body.GetProperty("slices").EnumerateArray().ToList();
        slices.Select(s => s.GetProperty("value").GetString()).Should().BeEquivalentTo("east", "west");
        slices.Sum(s => s.GetProperty("rows").GetInt32()).Should().Be(200);
        slices.Should().AllSatisfy(s => s.GetProperty("metrics").TryGetProperty("accuracy", out _).Should().BeTrue());
    }

    [Fact]
    public async Task AFileMissingTheModelsColumns_IsTheCallersError_NotAServerError()
    {
        var path = Path.Combine(_factory.TestProjectRoot, "datasets", "no-x.csv");
        await File.WriteAllLinesAsync(path, (await File.ReadAllLinesAsync(_withWeight)).Select(l => string.Join(',', l.Split(',').Skip(1))));

        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = path });

        status.Should().Be(HttpStatusCode.BadRequest, body.ToString());
        body.GetProperty("error").GetString().Should().Contain("Schema validation failed");
        body.GetProperty("missingColumns").EnumerateArray().Select(c => c.GetString()).Should().Contain("x");
    }

    [Fact]
    public async Task TheProjectsMetricScripts_AreComputedOverTheEvaluatedRows()
    {
        var scripts = Directory.CreateDirectory(Path.Combine(_factory.TestProjectRoot, ".mloop", "scripts", "metrics")).FullName;
        var script = Path.Combine(scripts, "RowCount.cs");
        await File.WriteAllTextAsync(script, """
            using System.Threading.Tasks;
            using MLoop.Extensibility.Metrics;
            public class RowCountMetric : IMLoopMetric
            {
                public string Name => "evaluated_rows";
                public string Description => "";
                public Task<MetricResult> CalculateAsync(MetricContext ctx)
                    => Task.FromResult(new MetricResult { Name = Name, Value = ((bool[])ctx.Labels).Length });
            }
            """);
        try
        {
            var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withoutWeight });

            status.Should().Be(HttpStatusCode.OK, body.ToString());
            body.GetProperty("metrics").GetProperty("evaluated_rows").GetDouble().Should().Be(200);
        }
        finally
        {
            File.Delete(script);
        }
    }

    [Fact]
    public async Task ATabSeparatedFile_IsReadAsATable_AndARelativePathIsTheProjects()
    {
        var tsv = Path.Combine(_factory.TestProjectRoot, "datasets", "weighted.tsv");
        await File.WriteAllLinesAsync(tsv, (await File.ReadAllLinesAsync(_withWeight)).Select(l => l.Replace(',', '\t')));

        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = Path.Combine("datasets", "weighted.tsv") });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        body.GetProperty("weightColumn").GetString().Should().Be("w");
        body.GetProperty("metrics").TryGetProperty("accuracy", out _).Should().BeTrue();
    }

    [Fact]
    public async Task AWeightedExperiment_OnAFileWithoutItsWeight_SaysTheMetricsAreUnweighted()
    {
        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withoutWeight });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        body.TryGetProperty("weightColumn", out _).Should().BeFalse("the API leaves null fields out");
        body.GetProperty("warnings")[0].GetString().Should().Contain("weighted by 'w'");
    }

    [Fact]
    public async Task Against_PairsTheTwoModelsOnTheSameRowsAndWeights()
    {
        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-002", testDataPath = _withWeight, against = "exp-001" });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        var comparison = body.GetProperty("comparison");
        comparison.GetProperty("against").GetString().Should().Be("exp-001");
        comparison.GetProperty("metrics").GetProperty("accuracy").GetDouble().Should().BeInRange(0, 1);
        var paired = comparison.GetProperty("paired");
        paired.GetProperty("measure").GetString().Should().Be("error_rate");
        paired.GetProperty("rows").GetInt32().Should().Be(200);
        paired.GetProperty("low").GetDouble().Should().BeLessThanOrEqualTo(paired.GetProperty("high").GetDouble());
        paired.GetProperty("verdict").GetString().Should().BeOneOf("CandidateBetter", "BaselineBetter", "Indistinguishable");
    }

    [Fact]
    public async Task AgainstItself_IsABadRequest_AndAgainstAnUnknownExperiment_IsNotFound()
    {
        var (self, selfBody) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withWeight, against = "exp-001" });
        var (missing, missingBody) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withWeight, against = "exp-404" });

        self.Should().Be(HttpStatusCode.BadRequest);
        selfBody.GetProperty("error").GetString().Should().Contain("the experiment being evaluated");
        missing.Should().Be(HttpStatusCode.NotFound);
        missingBody.GetProperty("error").GetString().Should().Contain("exp-404");
    }
}
