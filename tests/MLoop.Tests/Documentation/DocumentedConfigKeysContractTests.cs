using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The <c>mloop.yaml</c> examples in the documentation use only settings that are read.
/// </summary>
/// <remarks>
/// <c>mloop.yaml</c> is read leniently, so an example with a misspelled or retired key works — it just does nothing —
/// and a reader copies it. <see cref="ConfigKeyAudit"/> now says so for a user's file; this holds the documentation to
/// the same standard. Only blocks that are visibly mloop configuration are audited (a whole file or a model-level
/// fragment); a workflow or a Kubernetes manifest in the same guide is not.
/// </remarks>
public class DocumentedConfigKeysContractTests
{
    private static readonly Regex YamlBlock = new(@"```yaml\r?\n(?<body>.*?)```", RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly string[] ProjectKeys = ["project", "models", "data", "retention"];

    private static readonly string[] ModelKeys = ["prep", "columns", "gate", "training"];

    private static IEnumerable<(string File, int Index, string Body)> Blocks() =>
        new[] { "README.md", "docs" }
            .Select(p => Path.Combine(RepoSourceTree.RepoRoot, p))
            .SelectMany(p => Directory.Exists(p) ? Directory.EnumerateFiles(p, "*.md", SearchOption.AllDirectories) : File.Exists(p) ? [p] : [])
            .SelectMany(file => YamlBlock.Matches(File.ReadAllText(file)).Select((m, i) => (file, i, m.Groups["body"].Value)));

    /// <summary>The configuration type a block is, or null when it is some other YAML.</summary>
    private static Type? ConfigurationTypeOf(string body)
    {
        var first = body.Split('\n').Select(l => l.TrimEnd('\r')).FirstOrDefault(l => l.Length > 0 && !l.TrimStart().StartsWith('#'));
        var key = first is null || char.IsWhiteSpace(first[0]) ? null : first.Split(':')[0].Trim();
        if (key is null) return null;
        return ProjectKeys.Contains(key) ? typeof(MLoopConfig) : ModelKeys.Contains(key) ? typeof(ModelDefinition) : null;
    }

    [Fact]
    public void EveryConfigurationExample_UsesOnlySettingsThatAreRead()
    {
        var audited = 0;
        var offenders = new List<string>();
        foreach (var (file, index, body) in Blocks())
        {
            if (ConfigurationTypeOf(body) is not { } type)
                continue;
            audited++;
            foreach (var unknown in ConfigKeyAudit.Find(body, type))
                offenders.Add($"{Path.GetRelativePath(RepoSourceTree.RepoRoot, file)} (yaml block {index + 1}): {unknown.Describe()}");
        }

        Assert.True(audited >= 4, $"the scan found {audited} mloop.yaml examples — it should see the guide's model, gate, retention and prep ones");
        Assert.Empty(offenders);
    }

    [Fact]
    public void TheScan_RecognisesAConfigurationBlock_AndFlagsAMisspelledOne()
    {
        // The exact shape it polices, and a legitimate one that must stay quiet (a workflow is not mloop configuration).
        const string Misspelled = "models:\n  default:\n    task: regression\n    label: y\n    gate:\n      no_worse_than_prod: true\n";
        Assert.Equal(typeof(MLoopConfig), ConfigurationTypeOf(Misspelled));
        Assert.Single(ConfigKeyAudit.Find(Misspelled, typeof(MLoopConfig)));

        Assert.Null(ConfigurationTypeOf("# .github/workflows/train.yml\njobs:\n  train:\n    steps: []\n"));
        Assert.Equal(typeof(ModelDefinition), ConfigurationTypeOf("prep:\n  - type: explode\n    columns: [a]\n"));
        Assert.Empty(ConfigKeyAudit.Find("prep:\n  - type: explode\n    columns: [a]\n", typeof(ModelDefinition)));
    }
}
