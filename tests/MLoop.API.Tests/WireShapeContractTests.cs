using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.DataStore.Interfaces;

namespace MLoop.API.Tests;

/// <summary>
/// The field names and JSON kinds the HTTP API answers with, pinned against a recorded baseline
/// (<c>WireShapes.baseline.txt</c>). A consumer reads these fields by name — renaming one used to pass every
/// test here (the response is an anonymous object, and the hand-written assertions name the fields they happen to
/// care about), so the consumer learned of it from a published version.
///
/// <para>Each scenario calls an endpoint on a project seeded without training and records its answer's
/// <i>shape</i>: every path with the kinds seen at it (<c>string</c> · <c>number</c> · <c>boolean</c> ·
/// <c>object</c> · <c>array</c> · <c>null</c>), never a value. A path that is gone, renamed or retyped fails; so does a
/// new one — adding a field is additive, but it is deliberate: update the baseline and say so in the CHANGELOG.
/// A route the server maps that no scenario claims (and no exemption names) fails too, so a new endpoint cannot
/// join the surface unpinned.</para>
/// </summary>
public class WireShapeContractTests : IClassFixture<TestWebApplicationFactory>
{
    private const string BaselineFile = "WireShapes.baseline.txt";

    /// <summary>Set to a path: the test writes the shapes it observed there instead of failing, for a deliberate update.</summary>
    private const string WriteVariable = "MLOOP_WRITE_WIRE_SHAPES";

    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public WireShapeContractTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private sealed record Scenario(string Id, string Method, string Route, string Url, object? Body = null);

    /// <summary>Routes this test does not call, each with why and the class that pins that answer's shape instead.</summary>
    private static readonly Dictionary<string, (string Why, Type CoveredBy)> Exempt = new()
    {
        ["POST /predict"] = ("answers per task (table · forecast · image · detection) and needs a trained model", typeof(PredictResponseContractTests)),
        ["POST /evaluate"] = ("needs a trained model and a labeled file", typeof(EvaluateApiTests)),
        ["POST /train"] = ("starts a training job", typeof(TrainDefinitionApiTests)),
        ["POST /embed"] = ("needs an installed image encoder", typeof(EmbedApiTests)),
        ["HEAD /health"] = ("the probe's HEAD form: the same endpoint without a body", typeof(ApiIntegrationTests)),
    };

    private static List<Scenario> Scenarios() =>
    [
        new("health", "GET", "/health", "/health"),
        new("cache-stats", "GET", "/cache/stats", "/cache/stats"),
        new("cache-clear", "POST", "/cache/clear", "/cache/clear"),
        new("promote-ranks", "POST", "/promote", "/promote", new { experimentId = "exp-001", name = "ranks" }),
        new("promote-first", "POST", "/promote", "/promote", new { experimentId = "exp-001", name = "shape" }),
        new("promote-weaker", "POST", "/promote", "/promote", new { experimentId = "exp-002", name = "shape" }),
        new("promote-better", "POST", "/promote", "/promote", new { experimentId = "exp-003", name = "shape" }),
        new("promote-unknown", "POST", "/promote", "/promote", new { experimentId = "exp-099", name = "shape" }),
        new("promote-refused", "POST", "/promote", "/promote", new { experimentId = "exp-001", name = "gate" }),
        new("promote-no-id", "POST", "/promote", "/promote", new { name = "shape" }),
        new("info", "GET", "/info", "/info?name=shape"),
        new("info-none", "GET", "/info", "/info?name=nomodel"),
        new("models", "GET", "/models", "/models?name=shape"),
        new("promotions", "GET", "/promotions", "/promotions?name=shape"),
        new("experiments", "GET", "/experiments", "/experiments?name=shape"),
        new("experiment-detail", "GET", "/experiments/{id}", "/experiments/exp-001?name=shape"),
        new("experiment-missing", "GET", "/experiments/{id}", "/experiments/exp-099?name=shape"),
        new("compare", "GET", "/compare", "/compare?candidate=exp-003&baseline=exp-001&name=shape"),
        new("compare-missing", "GET", "/compare", "/compare?candidate=exp-099&baseline=exp-001&name=shape"),
        new("status", "GET", "/status", "/status"),
        new("trigger", "GET", "/trigger", "/trigger?name=shape"),
        new("drift", "GET", "/drift", "/drift?name=shape"),
        new("drift-window", "GET", "/drift", "/drift?name=shape&rows=1"),
        new("logs", "GET", "/logs", "/logs?name=shape"),
        new("logs-bad-date", "GET", "/logs", "/logs?name=shape&from=not-a-date"),
        new("feedback-add", "POST", "/feedback", "/feedback", new { predictionId = "{prediction}", actualValue = 3.5, source = "contract" }),
        new("feedback-unknown", "POST", "/feedback", "/feedback", new { predictionId = "no-such-prediction", actualValue = 1 }),
        new("feedback-list", "GET", "/feedback", "/feedback?name=shape"),
        new("samples", "POST", "/samples", "/samples", new { name = "shape", rows = 5 }),
        new("samples-bad-rows", "POST", "/samples", "/samples", new { name = "shape", rows = 0 }),
        new("jobs", "GET", "/jobs", "/jobs"),
        new("job-missing", "GET", "/jobs/{id}", "/jobs/no-such-job"),
        new("rollback", "POST", "/rollback", "/rollback", new { name = "shape" }),
        new("rollback-none", "POST", "/rollback", "/rollback", new { name = "nomodel" }),
    ];

    [Fact]
    public async Task EveryEndpointAnswersWithTheRecordedShape()
    {
        await SeedAsync();
        var predictionId = (await _factory.Services.GetRequiredService<IPredictionLogger>().GetLogsAsync("shape")).First().Id!;

        var observed = new List<(string Id, string Header, SortedDictionary<string, string> Shape)>();
        foreach (var scenario in Scenarios())
        {
            using var request = new HttpRequestMessage(new HttpMethod(scenario.Method), scenario.Url);
            if (scenario.Body is not null)
            {
                var json = JsonSerializer.Serialize(scenario.Body).Replace("{prediction}", predictionId);
                request.Content = new StringContent(json, Encoding.UTF8, "application/json");
            }
            var response = await _client.SendAsync(request);
            var text = await response.Content.ReadAsStringAsync();
            var shape = text.Length == 0 ? new SortedDictionary<string, string>() : WireShape.Describe(text);
            observed.Add((scenario.Id, $"{scenario.Method} {scenario.Route} -> {(int)response.StatusCode}"
                + (text.Length == 0 ? " (no body)" : ""), shape));
        }

        var rendered = WireShape.Render(observed.Select(o => (o.Id, o.Header, (IReadOnlyDictionary<string, string>)o.Shape)));
        var target = Environment.GetEnvironmentVariable(WriteVariable);
        if (!string.IsNullOrEmpty(target))
        {
            await File.WriteAllTextAsync(target, rendered);
            return;
        }

        var baselinePath = Path.Combine(AppContext.BaseDirectory, BaselineFile);
        File.Exists(baselinePath).Should().BeTrue($"{BaselineFile} is copied beside the test assembly");
        var baseline = WireShape.Parse(await File.ReadAllTextAsync(baselinePath));
        var differences = WireShape.Differences(baseline, WireShape.Parse(rendered));
        differences.Should().BeEmpty(
            "the answers' field names and kinds are a contract — a removed, renamed or retyped field breaks a consumer; " +
            "a new one is additive but deliberate: regenerate with " + WriteVariable + "=<path to tests/MLoop.API.Tests/" + BaselineFile + "> and say so in the CHANGELOG");
    }

    [Fact]
    public void EveryMappedRouteIsClaimedByAScenarioOrNamedInAnExemption()
    {
        var mapped = _factory.Services.GetServices<EndpointDataSource>()
            .SelectMany(source => source.Endpoints)
            .OfType<RouteEndpoint>()
            .Select(endpoint => (Pattern: endpoint.RoutePattern.RawText ?? "",
                Methods: endpoint.Metadata.GetMetadata<HttpMethodMetadata>()?.HttpMethods ?? []))
            .Where(route => route.Methods.Count > 0 && !route.Pattern.StartsWith("/swagger", StringComparison.Ordinal))
            .SelectMany(route => route.Methods.Select(method => $"{method} {route.Pattern}"))
            .ToList();

        var claimed = Scenarios().Select(s => $"{s.Method} {s.Route}").Concat(Exempt.Keys).ToHashSet(StringComparer.Ordinal);

        mapped.Should().NotBeEmpty("the scan has to see the server's routes — an empty list would pass every route");
        WireShape.Unclaimed(mapped, claimed).Should().BeEmpty(
            "a route no scenario calls has no pinned shape: add a scenario to " + nameof(WireShapeContractTests) +
            ", or name it in Exempt with why and the test class that pins it");
        mapped.Should().Contain("GET /compare", "the scan found the route the 1392 mutation renamed fields of");
    }

    [Fact]
    public void EveryExemptionNamesAClassThatExistsAndARouteThatIsMapped()
    {
        foreach (var (route, (why, coveredBy)) in Exempt)
        {
            why.Should().NotBeNullOrWhiteSpace(route);
            coveredBy.GetMethods().Any(m => m.GetCustomAttributes(typeof(FactAttribute), true).Length > 0 ||
                                            m.GetCustomAttributes(typeof(TheoryAttribute), true).Length > 0)
                .Should().BeTrue($"{coveredBy.Name} must hold tests — the exemption for {route} points at it");
        }
    }

    // — the guard asserts that it can see: the shapes it forbids, and one it must not refuse —

    [Fact]
    public void ARenamedFieldIsReportedAsOneRemovedAndOneAdded()
    {
        var before = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":1,"b":"x"}"""))]));
        var after = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":1,"c":"x"}"""))]));

        WireShape.Differences(before, after).Should().BeEquivalentTo(["s: removed $.b: string", "s: added $.c: string"]);
    }

    [Fact]
    public void ARetypedFieldIsReported()
    {
        var before = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":1}"""))]));
        var after = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":"1"}"""))]));

        WireShape.Differences(before, after).Should().NotBeEmpty();
    }

    [Fact]
    public void AChangedStatusIsReported()
    {
        var before = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":1}"""))]));
        var after = WireShape.Parse(WireShape.Render([("s", "GET /x -> 404", WireShape.Describe("""{"a":1}"""))]));

        WireShape.Differences(before, after).Should().NotBeEmpty();
    }

    [Fact]
    public void ADifferentValueOfTheSameKindIsNotADifference()
    {
        var before = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":1,"l":[{"k":"x"}]}"""))]));
        var after = WireShape.Parse(WireShape.Render([("s", "GET /x -> 200", WireShape.Describe("""{"a":2,"l":[{"k":"y"},{"k":"z"}]}"""))]));

        WireShape.Differences(before, after).Should().BeEmpty();
    }

    [Fact]
    public void ArrayElementsAreUnionedAndANullIsAKind()
    {
        var shape = WireShape.Describe("""{"l":[{"k":1},{"k":null,"m":true}]}""");

        shape["$.l[].k"].Should().Be("null|number");
        shape["$.l[].m"].Should().Be("boolean");
    }

    [Fact]
    public void AnUnclaimedRouteIsFoundAndAClaimedOneIsNot()
    {
        WireShape.Unclaimed(["GET /a", "GET /b"], new HashSet<string> { "GET /a" }).Should().Equal("GET /b");
    }

    private async Task SeedAsync()
    {
        var store = _factory.Services.GetRequiredService<IExperimentStore>();
        async Task Save(string model, string id, Dictionary<string, double> metrics, string optimized = "f1_score")
        {
            var path = store.GetExperimentPath(model, id);
            Directory.CreateDirectory(path);
            await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
            await store.SaveAsync(model, new ExperimentData
            {
                ModelName = model,
                ExperimentId = id,
                Timestamp = new DateTime(2026, 11, 12, 9, 0, 0, DateTimeKind.Utc),
                Status = "Completed",
                Task = "binary-classification",
                Config = new ExperimentConfig
                {
                    DataFile = "train.csv",
                    LabelColumn = "y",
                    Metric = optimized,
                    TimeLimitSeconds = 60,
                    TestSplit = 0.2,
                },
                Metrics = metrics,
            }, CancellationToken.None);
        }

        await Save("shape", "exp-001", new() { ["accuracy"] = 0.91, ["f1_score"] = 0.88, ["negative_recall"] = 0.94, ["auc"] = 0.95 });
        await Save("shape", "exp-002", new() { ["accuracy"] = 0.575, ["f1_score"] = 0.61, ["negative_recall"] = 0.5, ["auc"] = 0.57 });
        await Save("shape", "exp-003", new() { ["accuracy"] = 0.95, ["f1_score"] = 0.92, ["negative_recall"] = 0.96, ["auc"] = 0.97 });
        // A model the quality gate refuses (one class answered to every row), so the refusal's answer has a shape here.
        await Save("gate", "exp-001", new() { ["accuracy"] = 0.993, ["f1_score"] = 0, ["negative_recall"] = 1, ["auc"] = 0.95 });
        // A model that ranks well but answers one class at its default threshold: promoted, with a quality warning.
        await Save("ranks", "exp-001", new() { ["accuracy"] = 0.969, ["f1_score"] = 0, ["negative_recall"] = 1, ["auc"] = 0.968 }, optimized: "auc");

        await _factory.Services.GetRequiredService<IPredictionLogger>().LogPredictionAsync(
            "shape", "exp-001", new Dictionary<string, object> { ["x"] = 1.0, ["kind"] = "a" }, true, 0.9);
    }
}

/// <summary>The shape of a JSON answer — path → kinds — and its baseline text form.</summary>
internal static class WireShape
{
    public static SortedDictionary<string, string> Describe(string json)
    {
        using var document = JsonDocument.Parse(json);
        var kinds = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        Walk(document.RootElement, "$", kinds);
        return new SortedDictionary<string, string>(
            kinds.ToDictionary(pair => pair.Key, pair => string.Join('|', pair.Value)), StringComparer.Ordinal);
    }

    private static void Walk(JsonElement element, string path, SortedDictionary<string, SortedSet<string>> kinds)
    {
        void Add(string kind)
        {
            if (!kinds.TryGetValue(path, out var set)) kinds[path] = set = new SortedSet<string>(StringComparer.Ordinal);
            set.Add(kind);
        }

        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                Add("object");
                foreach (var property in element.EnumerateObject()) Walk(property.Value, path + "." + property.Name, kinds);
                break;
            case JsonValueKind.Array:
                Add("array");
                foreach (var item in element.EnumerateArray()) Walk(item, path + "[]", kinds);
                break;
            case JsonValueKind.String: Add("string"); break;
            case JsonValueKind.Number: Add("number"); break;
            case JsonValueKind.True:
            case JsonValueKind.False: Add("boolean"); break;
            default: Add("null"); break;
        }
    }

    /// <summary>One block per scenario: <c>## id | header</c>, then <c>path: kinds</c> lines.</summary>
    public static string Render(IEnumerable<(string Id, string Header, IReadOnlyDictionary<string, string> Shape)> blocks)
    {
        var text = new StringBuilder();
        foreach (var (id, header, shape) in blocks)
        {
            text.Append("## ").Append(id).Append(" | ").Append(header).Append('\n');
            foreach (var (path, kinds) in shape.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                text.Append(path).Append(": ").Append(kinds).Append('\n');
        }
        return text.ToString();
    }

    public static Dictionary<string, HashSet<string>> Parse(string text)
    {
        var blocks = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        HashSet<string>? current = null;
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0) continue;
            if (line.StartsWith("## ", StringComparison.Ordinal))
            {
                var id = line[3..].Split(" | ")[0];
                current = blocks[id] = [line];
            }
            else current?.Add(line);
        }
        return blocks;
    }

    public static List<string> Differences(Dictionary<string, HashSet<string>> baseline, Dictionary<string, HashSet<string>> observed)
    {
        var differences = new List<string>();
        foreach (var id in baseline.Keys.Union(observed.Keys).OrderBy(k => k, StringComparer.Ordinal))
        {
            if (!baseline.TryGetValue(id, out var was)) { differences.Add($"{id}: scenario is not in the baseline"); continue; }
            if (!observed.TryGetValue(id, out var now)) { differences.Add($"{id}: scenario was not observed"); continue; }
            differences.AddRange(was.Except(now).OrderBy(l => l, StringComparer.Ordinal)
                .Select(line => line.StartsWith("## ", StringComparison.Ordinal) ? $"{id}: status changed (was {line})" : $"{id}: removed {line}"));
            differences.AddRange(now.Except(was).OrderBy(l => l, StringComparer.Ordinal)
                .Where(line => !line.StartsWith("## ", StringComparison.Ordinal)).Select(line => $"{id}: added {line}"));
        }
        return differences;
    }

    public static List<string> Unclaimed(IEnumerable<string> mapped, ISet<string> claimed) =>
        mapped.Where(route => !claimed.Contains(route)).Distinct().OrderBy(r => r, StringComparer.Ordinal).ToList();
}
