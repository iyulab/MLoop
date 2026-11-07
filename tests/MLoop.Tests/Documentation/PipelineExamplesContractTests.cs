using System.Text.RegularExpressions;
using MLoop.CLI;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using YamlDotNet.RepresentationModel;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The reference pipeline in <c>docs/examples/pipeline</c> - a workflow, a cron script and the page that explains them - keeps
/// its promises: the workflow is YAML, every <c>mloop</c> command in the three files parses against the real command tree,
/// and the JSON fields the workflow reads with <c>jq</c> are the ones the tool writes.
/// </summary>
/// <remarks>
/// The README and the guide hold their commands to the grammar in <see cref="ReadmeCommandContractTests"/>; these files live
/// elsewhere and wrap their commands in YAML and shell, so a command could be renamed or lose an option and the example
/// would go on being copied. The syntax is what this can check - that the example runs end to end is shown by running it,
/// and the part a parse cannot see is the shape of the JSON a step reads, which is pinned against the emitter below.
/// </remarks>
public class PipelineExamplesContractTests
{
    private static readonly string Directory = Path.Combine(RepoSourceTree.RepoRoot, "docs", "examples", "pipeline");

    private static readonly string[] Files = ["README.md", "retrain-and-promote.yml", "watch-production.sh"];

    private static string[] Lines(string file) => File.ReadAllLines(Path.Combine(Directory, file));

    public static IEnumerable<object[]> Commands() =>
        Files
            .SelectMany(file => ReadmeCommandContractTests.ExtractCommands(Lines(file)).Select(command => (file, command)))
            .DistinctBy(x => x.command)
            .Select(x => new object[] { x.file, x.command });

    [Theory]
    [MemberData(nameof(Commands))]
    public void Example_parses_without_error(string file, string commandLine)
    {
        var args = ReadmeCommandContractTests.Tokenize(commandLine).Skip(1).ToArray();
        var result = Program.BuildRootCommand().Parse(args);

        Assert.True(result.Errors.Count == 0,
            $"{file} example \"{commandLine}\" failed to parse: " + string.Join("; ", result.Errors.Select(e => e.Message)));
    }

    [Fact]
    public void Every_file_shows_commands_and_the_scan_sees_them()
    {
        // A file reshaped so that no line begins with `mloop ` would leave every case above vacuously passing.
        foreach (var file in Files)
            Assert.True(ReadmeCommandContractTests.ExtractCommands(Lines(file)).Count > 0, $"{file}: no `mloop ...` command lines found.");

        // The workflow runs the stages the page names, in the order it names them.
        var workflow = string.Join('\n', Lines("retrain-and-promote.yml"));
        var order = new[] { "mloop validate", "mloop train", "mloop evaluate", "mloop promote" }.Select(c => workflow.IndexOf(c, StringComparison.Ordinal)).ToArray();
        Assert.All(order, i => Assert.True(i >= 0));
        Assert.Equal(order.OrderBy(i => i), order);
    }

    [Fact]
    public void The_check_catches_a_command_the_tool_does_not_have()
    {
        // The shape it polices: an option that was renamed away.
        var broken = Program.BuildRootCommand().Parse(ReadmeCommandContractTests.Tokenize("mloop promote exp-001 --who ci").Skip(1).ToArray());
        Assert.NotEmpty(broken.Errors);
    }

    [Fact]
    public void The_workflow_is_yaml_with_a_job_that_has_steps()
    {
        var stream = new YamlStream();
        using (var reader = new StringReader(string.Join('\n', Lines("retrain-and-promote.yml"))))
            stream.Load(reader);

        var root = Assert.IsType<YamlMappingNode>(stream.Documents[0].RootNode);
        var jobs = Assert.IsType<YamlMappingNode>(root.Children[new YamlScalarNode("jobs")]);
        var job = Assert.IsType<YamlMappingNode>(jobs.Children.Values.First());
        var steps = Assert.IsType<YamlSequenceNode>(job.Children[new YamlScalarNode("steps")]);
        Assert.True(steps.Children.Count >= 5);
    }

    [Fact]
    public void The_events_the_workflow_reads_with_jq_are_the_ones_train_json_writes()
    {
        // `select(.event == "result") | .experimentId`: the stage that names the candidate for the stages after it.
        var workflow = string.Join('\n', Lines("retrain-and-promote.yml"));
        var match = Regex.Match(workflow, @"select\(\.event == ""(?<event>\w+)""\) \| \.(?<field>\w+)");
        Assert.True(match.Success, "the workflow no longer reads the candidate's id from the train events");

        var sink = new StringWriter();
        new TrainJsonEmitter(sink).Result(new TrainingResult
        {
            ExperimentId = "exp-007",
            Trainer = TrainerDescriptor.Of("LightGbmBinary"),
            Metrics = new Dictionary<string, double> { ["accuracy"] = 0.9 },
            TrainingTimeSeconds = 1,
            ModelPath = "model.zip",
        }, "default");

        var line = System.Text.Json.JsonDocument.Parse(sink.ToString().Trim()).RootElement;
        Assert.Equal(match.Groups["event"].Value, line.GetProperty("event").GetString());
        Assert.Equal("exp-007", line.GetProperty(match.Groups["field"].Value).GetString());
    }

    [Fact]
    public void The_fields_the_cron_script_and_the_comparison_step_read_are_the_ones_the_tool_writes()
    {
        // `shouldRetrain`, `conditions[].isMet` / `.name` / `.details`: the RetrainingTrigger result; `.comparison.paired`: `evaluate --against`.
        var script = string.Join('\n', Lines("watch-production.sh"));
        foreach (var field in Regex.Matches(script, @"\.(shouldRetrain|isMet|name|details)\b").Select(m => m.Groups[1].Value).Distinct())
            Assert.True(
                new[] { typeof(MLoop.Ops.Interfaces.TriggerEvaluation), typeof(MLoop.Ops.Interfaces.ConditionResult), typeof(MLoop.Ops.Interfaces.RetrainingCondition) }
                    .Any(t => t.GetProperties().Any(p => string.Equals(p.Name, field, StringComparison.OrdinalIgnoreCase))),
                $"watch-production.sh reads .{field}, which the trigger result no longer has");
    }
}
