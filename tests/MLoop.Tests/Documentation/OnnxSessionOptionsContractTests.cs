using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// An ONNX Runtime session is opened with the options <c>OnnxSessionOptions</c> makes. Its intra-op threads
/// spin when idle by default, which kept a serving process on the cores for a moment after every answer —
/// taking them from the caller on the same machine that uses the answer at once. A session opened with its own
/// <c>new SessionOptions</c> goes back to that default without anything saying so.
/// </summary>
public class OnnxSessionOptionsContractTests
{
    private static readonly Regex OwnOptions = new(@"new\s+SessionOptions\b|\bSessionOptions\s+\w+\s*=\s*new\s*\(", RegexOptions.Compiled);

    private static bool IsDocumentation(string line) => line.TrimStart().StartsWith("//", StringComparison.Ordinal);

    private static IEnumerable<string> Offenders(IEnumerable<(string File, string[] Lines)> sources) =>
        from source in sources
        where Path.GetFileName(source.File) != "OnnxSessionOptions.cs"
        from line in source.Lines.Select((Text, Number) => (Text, Number))
        where !IsDocumentation(line.Text) && OwnOptions.IsMatch(line.Text)
        select $"{source.File}:{line.Number + 1}";

    [Fact]
    public void ProductCodeMakesSessionOptionsInOnePlace()
    {
        var sources = RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Select(f => (File: RepoSourceTree.RelativeToRoot(f), Lines: File.ReadAllLines(f))).ToList();
        // It read the tree: the authority is among what it scanned, and it does make the options.
        Assert.Contains(sources, s => Path.GetFileName(s.File) == "OnnxSessionOptions.cs" && s.Lines.Any(OwnOptions.IsMatch));
        var offenders = Offenders(sources).ToList();

        Assert.True(offenders.Count == 0,
            "Open an ONNX Runtime session with MLoop.Core.Embedding.OnnxSessionOptions.Create(), not a SessionOptions of its own:" +
            Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The scan can see: it finds the shape it forbids, and lets the authority itself and a comment pass.</summary>
    [Fact]
    public void TheScanSeesASessionOptionsOfItsOwn()
    {
        var forbidden = ("src/X.cs", new[] { "        using var options = new SessionOptions { LogSeverityLevel = level };" });
        var targetTyped = ("src/Z.cs", new[] { "        SessionOptions options = new();" });
        var authority = ("src/MLoop.Core/Embedding/OnnxSessionOptions.cs", new[] { "        var options = new SessionOptions { };" });
        var comment = ("src/Y.cs", new[] { "        // never new SessionOptions here" });

        Assert.Equal(["src/X.cs:1", "src/Z.cs:1"], Offenders([forbidden, targetTyped, authority, comment]));
    }
}
