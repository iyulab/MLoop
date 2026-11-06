using System.Net;
using System.Net.Http.Json;
using System.Reflection;
using System.Text;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;

namespace MLoop.API.Tests;

/// <summary>
/// A model name a caller supplies becomes <c>models/&lt;name&gt;/…</c> and <c>.mloop/logs/&lt;name&gt;/</c>, so every endpoint that takes
/// one refuses a name that is not a valid directory name — <c>400</c>, with the reason — and none of them leaves <c>models/</c>.
/// The earlier sweep (the earlier query-value sweep) only asked "never a 500", which a <c>200</c> from a directory outside the
/// project also satisfies; this one asks for the refusal itself, on the read endpoints, on the ones that take the name in a
/// body, and on <c>POST /train</c>, which wrote an experiment tree wherever <c>..</c> led.
/// </summary>
public class ModelNameGateApiTests : IClassFixture<TestWebApplicationFactory>
{
    private const string Refusal = "Invalid model name";

    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ModelNameGateApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
        // A file that exists, so POST /train gets as far as the name: a request refused for a missing file says nothing about the gate.
        File.WriteAllText(Path.Combine(factory.TestProjectRoot, "datasets", PresentFile), "x,y\n1,0\n2,1\n3,0\n4,1\n");
    }

    private const string PresentFile = "t.csv";

    /// <summary>The names that must be refused: parents, separators of both kinds, a reserved word, a case the rule does not allow.</summary>
    public static readonly string[] BadNames =
        ["..", "../../other", "default/../../x", "a/b", "a\\b", "/etc", "C:\\x", "staging", "my_model", "x"];

    /// <summary>Every endpoint that takes the model name in its query string, as <c>METHOD path?query-with-{name}</c>.</summary>
    private static readonly string[] QueryEndpoints =
    [
        "GET /info?name={0}", "GET /models?name={0}", "GET /experiments?name={0}", "GET /experiments/exp-001?name={0}",
        "GET /logs?name={0}", "GET /feedback?name={0}", "GET /drift?name={0}", "GET /trigger?name={0}",
        "GET /promotions?name={0}", "GET /compare?candidate=exp-001&baseline=exp-002&name={0}", "POST /predict?name={0}",
    ];

    /// <summary>Every endpoint that takes it in the JSON body, with a body that is otherwise acceptable.</summary>
    private static readonly string[] BodyEndpoints =
    [
        "POST /promote", "POST /rollback", "POST /samples", "POST /evaluate", "POST /train",
    ];

    private static string BodyFor(string route, string name, string dataFile = "datasets/" + PresentFile)
    {
        var quoted = System.Text.Json.JsonSerializer.Serialize(name);
        return route switch
        {
            "POST /promote" => $"{{\"experimentId\":\"exp-001\",\"name\":{quoted}}}",
            "POST /evaluate" => $"{{\"testDataPath\":\"{dataFile}\",\"experimentId\":\"exp-001\",\"name\":{quoted}}}",
            "POST /train" => $"{{\"dataFile\":\"{dataFile}\",\"labelColumn\":\"y\",\"task\":\"binary-classification\",\"name\":{quoted}}}",
            _ => $"{{\"name\":{quoted}}}",
        };
    }

    public static TheoryData<string, string> Requests()
    {
        var data = new TheoryData<string, string>();
        foreach (var name in BadNames)
        {
            foreach (var endpoint in QueryEndpoints.Concat(BodyEndpoints))
                data.Add(endpoint, name);
        }
        return data;
    }

    private async Task<HttpResponseMessage> SendAsync(string endpoint, string name, string dataFile = "datasets/" + PresentFile)
    {
        var space = endpoint.IndexOf(' ');
        var method = new HttpMethod(endpoint[..space]);
        var target = endpoint[(space + 1)..];
        var request = target.Contains("{0}")
            ? new HttpRequestMessage(method, string.Format(target, Uri.EscapeDataString(name)))
            : new HttpRequestMessage(method, target);
        if (method == HttpMethod.Post)
            request.Content = new StringContent(target.Contains("{0}") ? "{}" : BodyFor(endpoint, name, dataFile), Encoding.UTF8, "application/json");
        return await _client.SendAsync(request);
    }

    [Theory]
    [MemberData(nameof(Requests))]
    public async Task AModelNameThatIsNotADirectoryName_IsRefusedBeforeAnythingIsRead(string endpoint, string name)
    {
        var response = await SendAsync(endpoint, name);
        var body = await response.Content.ReadAsStringAsync();

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest, $"{endpoint} with name '{name}' answered {(int)response.StatusCode}: {body}");
        body.Should().Contain(Refusal, $"{endpoint} refused '{name}' for another reason than the name: {body}");
    }

    /// <summary>The companion the guard needs: the same endpoints, a name that is valid. If the sweep above passed because every
    /// request is refused for any reason, this fails; if the refusal were a blanket 400, this fails too.</summary>
    [Theory]
    [InlineData("default")]
    [InlineData("my-model")]
    [InlineData("MyModel")]
    public async Task AValidModelName_IsNotRefusedAsAName(string name)
    {
        foreach (var endpoint in QueryEndpoints.Concat(BodyEndpoints))
        {
            var response = await SendAsync(endpoint, name, dataFile: "datasets/absent.csv");
            var body = await response.Content.ReadAsStringAsync();

            body.Should().NotContain(Refusal, $"{endpoint} refused the valid name '{name}': {body}");
        }
    }

    /// <summary>The routes that take a name are the routes this class sweeps. A new endpoint with a <c>name</c> parameter or a body field
    /// <c>Name</c> that is not listed above is a new door the sweep cannot see — this fails until it is added.</summary>
    [Fact]
    public void EveryEndpointThatTakesAModelName_IsInTheSweep()
    {
        var found = _factory.Services.GetRequiredService<EndpointDataSource>().Endpoints
            .OfType<RouteEndpoint>()
            .Where(TakesAModelName)
            .Select(e => $"{e.Metadata.GetMetadata<HttpMethodMetadata>()!.HttpMethods.First()} /{e.RoutePattern.RawText!.TrimStart('/')}")
            .ToHashSet();

        var swept = QueryEndpoints.Concat(BodyEndpoints)
            .Select(e => e.Contains('?') ? e[..e.IndexOf('?')] : e)
            .ToHashSet();

        found.Should().NotBeEmpty("the scan must see the endpoints it polices");
        found.Should().Contain("GET /info").And.Contain("POST /train", "known doors the scan has to find");
        // {id} in a route is written out in the sweep above.
        swept = swept.Select(s => s.Replace("/exp-001", "/{id}")).ToHashSet();
        found.Except(swept).Should().BeEmpty("these endpoints take a model name but the sweep does not send them one");
    }

    private static bool TakesAModelName(RouteEndpoint endpoint)
    {
        var handler = endpoint.Metadata.GetMetadata<MethodInfo>();
        if (handler is null)
            return false;
        return handler.GetParameters().Any(p =>
            (p.Name == "name" && p.ParameterType == typeof(string))
            || (p.GetCustomAttribute<FromBodyAttribute>() is not null
                && p.ParameterType.GetProperty("Name")?.PropertyType == typeof(string)));
    }

    /// <summary>The exact request that wrote outside the project on the published 0.49.0, and the one that read outside it.</summary>
    [Fact]
    public async Task ANameWithParents_NeitherReadsNorWritesOutsideTheProject()
    {
        var token = Guid.NewGuid().ToString("N");
        var sibling = Path.Combine(Path.GetTempPath(), $"mloop-sibling-{token}");
        Directory.CreateDirectory(Path.Combine(sibling, "production"));
        await File.WriteAllTextAsync(Path.Combine(sibling, "production", "metadata.json"), "{\"experimentId\":\"exp-001\"}");
        var written = Path.Combine(Path.GetTempPath(), $"mloop-written-{token}");
        try
        {
            // models/<name> is <project>/models/../../<x>, which is the temp directory's child.
            var read = await _client.GetAsync($"/info?name={Uri.EscapeDataString($"../../mloop-sibling-{token}")}");
            read.StatusCode.Should().Be(HttpStatusCode.BadRequest);

            var train = await _client.PostAsJsonAsync("/train", new
            {
                dataFile = "datasets/" + PresentFile, labelColumn = "y", task = "binary-classification", name = $"../../mloop-written-{token}",
            });
            train.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            Directory.Exists(written).Should().BeFalse("a refused request must not create the directory it named");
        }
        finally
        {
            foreach (var dir in new[] { sibling, written })
            {
                if (Directory.Exists(dir))
                    Directory.Delete(dir, recursive: true);
            }
        }
    }
}
