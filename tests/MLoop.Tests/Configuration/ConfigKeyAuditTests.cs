using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Text;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.Tests.Configuration;

/// <summary>
/// Keys in <c>mloop.yaml</c> that nothing reads. The file is read leniently, so a misspelled setting used to be
/// dropped without a word — for the promotion gate that meant a rule that was never judged, or a limit that
/// read as 0% and refused every model. Each test that expects a key to be reported has a partner that a correct
/// file is reported clean, so a walk that stopped seeing keys cannot pass as "no typos".
/// </summary>
public class ConfigKeyAuditTests
{
    private const string Typos = """
        project: demo
        gat:
          data: datasets/test.csv
        models:
          default:
            task: multiclass-classification
            label: status
            training:
              time_limit_secs: 5
              metric: auto
            gate:
              data: datasets/test.csv
              no_worse_than_prod: true
              limits:
                - actual: [fail]
                  predicted: [ok]
                  maxx: 0.02
        """;

    private const string Clean = """
        project: demo
        retention:
          days: 30
        models:
          Any Model Name:
            task: multiclass-classification
            label: status
            weight: w
            training:
              time_limit_seconds: 5
              metric: auto
              test_split: 0.2
            columns:
              "Odd Column Name":
                type: ignore
            prep:
              - type: normalize
                method: min-max
                columns: [age, score]
            gate:
              data: datasets/test.csv
              no_worse_than_production: true
              slice_by: line
              limits:
                - actual: [fail]
                  predicted: [ok]
                  max: 0.02
        data:
          train: datasets/train.csv
        """;

    [Fact]
    public void ATypoAtEveryDepth_IsReportedWithItsPathAndTheNearestSetting()
    {
        var unknown = ConfigKeyAudit.Find<MLoopConfig>(Typos).ToDictionary(u => u.Path, u => u.Suggestion);

        Assert.Equal(4, unknown.Count);
        Assert.Null(unknown["gat"]);   // three letters, two edits from the nearest setting: a slip too wide to name one
        Assert.Equal("time_limit_seconds", unknown["models.default.training.time_limit_secs"]);
        Assert.Equal("no_worse_than_production", unknown["models.default.gate.no_worse_than_prod"]);
        Assert.Equal("max", unknown["models.default.gate.limits[0].maxx"]);
    }

    [Fact]
    public void ACorrectFile_IsReportedClean_AndItsFreeFormNamesAreNotKeys()
    {
        // The partner of the test above: model names and column names are the user's own, and a list is walked item by item.
        Assert.Empty(ConfigKeyAudit.Find<MLoopConfig>(Clean));
    }

    [Fact]
    public void AnyKeyTheConfigurationSerializes_IsAKeyItReads()
    {
        // The audit and the reader share one list of settings: whatever the types write out, they read back.
        var config = new MLoopConfig
        {
            Project = "demo",
            Retention = new RetentionSettings { Days = 7 },
            Data = new DataSettings { Train = "a.csv", Test = "b.csv" },
            Models =
            {
                ["default"] = new ModelDefinition
                {
                    Task = "binary-classification",
                    Label = "y",
                    Weight = "w",
                    Training = new TrainingSettings { TimeLimitSeconds = 5, Metric = "auto", TestSplit = 0.2 },
                    Gate = new GatePolicy
                    {
                        Data = "g.csv", NoWorseThanProduction = true, SliceBy = "s",
                        Limits = [new GateLimit { Actual = ["a"], Predicted = ["b"], Max = 0.1 }],
                    },
                },
            },
        };
        var yaml = new SerializerBuilder().WithNamingConvention(UnderscoredNamingConvention.Instance)
            .ConfigureDefaultValuesHandling(DefaultValuesHandling.OmitNull).Build().Serialize(config);

        Assert.Empty(ConfigKeyAudit.Find<MLoopConfig>(yaml));
        Assert.Contains("no_worse_than_production", yaml);   // the serialized file really does carry the nested settings it is judged on
    }

    [Fact]
    public void TheFileInitWrites_IsReportedClean_ForEveryTask()
    {
        foreach (var task in TaskTypes.All)
        {
            var yaml = InitCommand.GetYamlTemplate("demo", task, "default", "label");
            Assert.Empty(ConfigKeyAudit.Find<MLoopConfig>(yaml));
        }
    }

    [Fact]
    public async Task ALoadSaysWhatItDropped_OncePerFile()
    {
        var root = Path.Combine(Path.GetTempPath(), "mloop_keyaudit_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllText(Path.Combine(root, "mloop.yaml"), Typos);
            var loader = new ConfigLoader(new FileSystemManager(), new FixedRoot(root));

            using var first = WarningConsole.Collect();
            await loader.LoadUserConfigAsync();
            await loader.LoadUserConfigAsync();

            var said = first.Warnings.Where(w => w.Contains("not a setting mloop reads")).ToList();
            Assert.Equal(4, said.Count);                                  // four keys, each once although the file was loaded twice
            Assert.Contains(said, w => w.Contains("Did you mean 'max'?"));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task TheOldSingleModelFormat_IsAuditedAgainstItsOwnKeys()
    {
        var root = Path.Combine(Path.GetTempPath(), "mloop_keyaudit_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var loader = new ConfigLoader(new FileSystemManager(), new FixedRoot(root));
            File.WriteAllText(Path.Combine(root, "mloop.yaml"), "project_name: p\ntask: regression\nlabel_colum: y\n");

            var unknown = Assert.Single(await loader.FindUnknownKeysAsync());
            Assert.Equal("label_colum", unknown.Path);
            Assert.Equal("label_column", unknown.Suggestion);

            File.WriteAllText(Path.Combine(root, "mloop.yaml"), "project_name: p\ntask: regression\nlabel_column: y\n");
            Assert.Empty(await loader.FindUnknownKeysAsync());   // the partner: the correct old file is clean
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ALimitWithoutMax_IsAnErrorInValidate_AndRefusesAtPromotion()
    {
        var gate = new GatePolicy { Data = "g.csv", Limits = [new GateLimit { Actual = ["fail"], Predicted = ["ok"] }] };
        var errors = new List<ValidateCommand.ValidationError>();
        ValidateCommand.ValidateGatePolicy("models.default.gate", gate, "multiclass-classification", errors, new List<ValidateCommand.ValidationWarning>());
        Assert.Contains(errors, e => e.Path == "models.default.gate.limits[0].max");

        var counts = new ConfusionCounts(["fail", "ok"], [[10, 0], [0, 10]]);   // nothing is wrong on these rows
        var reasons = GatePolicyCheck.Exceeded(gate.Limits!, counts, scope: null).ToList();
        Assert.Single(reasons);
        Assert.Contains("has no 'max'", reasons[0]);

        // And a limit that names its max, on the same perfect rows, refuses nothing.
        Assert.Empty(GatePolicyCheck.Exceeded([new GateLimit { Actual = ["fail"], Predicted = ["ok"], Max = 0 }], counts, scope: null));
    }

    [Fact]
    public void ANearName_IsASuggestion_AndAFarOneIsNot()
    {
        Assert.Equal("max", EditDistance.Nearest("maxx", ["max", "data"]));
        Assert.Null(EditDistance.Nearest("gate_policy", ["max", "data"]));
        Assert.Null(EditDistance.Nearest("x", ["y"]));   // a one-letter name is within two edits of everything
        Assert.Equal("no_worse_than_production", EditDistance.Nearest("no_worse_than_prod", ["no_worse_than_production", "weight"]));   // an abbreviation
        Assert.Null(EditDistance.Nearest("ti", ["time_limit_seconds", "timeout"]));   // too short to be an abbreviation of anything
        Assert.Null(EditDistance.Nearest("time", ["time_limit_seconds", "time_budget"]));   // two names begin so: not a single meaning
    }

    private sealed class FixedRoot(string root) : IProjectDiscovery
    {
        public string FindRoot() => root;
        public string FindRoot(string startingDirectory) => root;
        public bool IsProjectRoot(string path) => path == root;
        public void EnsureProjectRoot() { }
        public string GetMLoopDirectory(string projectRoot) => Path.Combine(root, ".mloop");
    }
}
