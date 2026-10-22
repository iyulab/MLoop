using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Every relative link in the published Markdown points at a file or folder that exists.
/// </summary>
/// <remarks>
/// Documents are deleted and renamed by the same commits that change the code, and the links into
/// them are in other files nobody opens in that commit. Eight such links had piled up — to guides
/// that were never written, to a plan that was deleted, to a notes folder that never left one
/// machine — before anything looked. A reader who follows one learns nothing except that the
/// documentation is not maintained.
/// </remarks>
public class DocumentationLinkContractTests
{
    /// <summary>
    /// A Markdown link target: <c>[text](target)</c> or <c>[text](target#anchor)</c>. Absolute URLs
    /// (<c>https:</c>, <c>mailto:</c>) and in-page anchors are not file links and are left alone.
    /// </summary>
    private static readonly Regex Link = new(@"\]\((?<target>[^)#\s]+)(?:#[^)\s]*)?\)", RegexOptions.Compiled);

    [Fact]
    public void EveryRelativeLinkInTheDocumentationResolves()
    {
        var markdown = RepoSourceTree.PublishedTextFiles()
            .Where(f => f.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
            .ToList();
        var checkedLinks = 0;
        var dead = new List<string>();

        foreach (var file in markdown)
        {
            var lines = File.ReadAllLines(file);
            for (var i = 0; i < lines.Length; i++)
            {
                foreach (var target in RelativeTargets(lines[i]))
                {
                    checkedLinks++;
                    var resolved = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, target));
                    if (!File.Exists(resolved) && !Directory.Exists(resolved))
                        dead.Add($"{RepoSourceTree.RelativeToRoot(file)}:{i + 1}  {target}");
                }
            }
        }

        // The scan reaches the links it is about: the README alone links the guide and the architecture.
        Assert.True(checkedLinks > 20, $"only {checkedLinks} relative links were found — the scan no longer sees the documentation");
        Assert.True(dead.Count == 0, "Relative links to nothing:\n" + string.Join('\n', dead));
    }

    [Theory]
    [InlineData("see [the guide](docs/GUIDE.md)", "docs/GUIDE.md")]
    [InlineData("[exit codes](../README.md#exit-codes)", "../README.md")]
    [InlineData("[a folder](examples/)", "examples/")]
    public void TheScanSeesARelativeLink(string line, string expected)
    {
        Assert.Equal([expected], RelativeTargets(line));
    }

    [Theory]
    [InlineData("[site](https://github.com/iyulab/MLoop)")]
    [InlineData("[mail](mailto:someone@example.com)")]
    [InlineData("[below](#exit-codes)")]
    public void TheScanLeavesAbsoluteUrlsAndAnchorsAlone(string line)
    {
        Assert.Empty(RelativeTargets(line));
    }

    private static IEnumerable<string> RelativeTargets(string line) =>
        Link.Matches(line)
            .Select(m => m.Groups["target"].Value)
            .Where(t => !Regex.IsMatch(t, "^[a-zA-Z][a-zA-Z0-9+.-]*:"));
}
