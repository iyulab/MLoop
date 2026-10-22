using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Every text loader product code builds reads an empty number as missing (<c>MissingNumbers</c>):
/// each <c>new TextLoader.Options { … }</c> sets <c>MissingRealsAsNaNs = true</c>, and no loader is built
/// through an overload that cannot (<c>LoadFromTextFile</c>). ML.NET's default reads the field as 0, so one
/// loader left at the default trains or predicts a dropped-out reading as a reading of zero.
/// </summary>
public class MissingNumberLoadingContractTests
{
    private static readonly Regex OptionsInitializer = new(@"new\s+(?:Microsoft\.ML\.Data\.)?TextLoader\.Options\s*(?:\(\s*\))?\s*\{", RegexOptions.Compiled);
    private static readonly Regex Overload = new(@"\.LoadFromTextFile\s*[<(]", RegexOptions.Compiled);
    private const string Setting = "MissingRealsAsNaNs = true";

    private static IEnumerable<string> Offenders(IEnumerable<(string File, string Text)> sources)
    {
        foreach (var (file, text) in sources)
        {
            var code = string.Join('\n', text.Split('\n').Select(l => l.TrimStart().StartsWith("//", StringComparison.Ordinal) ? "" : l));
            foreach (Match m in OptionsInitializer.Matches(code))
            {
                var body = Initializer(code, m.Index + m.Length - 1);
                if (!body.Contains(Setting, StringComparison.Ordinal))
                    yield return $"{file}:{LineOf(code, m.Index)}";
            }
            foreach (Match m in Overload.Matches(code))
                yield return $"{file}:{LineOf(code, m.Index)}";
        }
    }

    // The text from the initializer's opening brace to the brace that closes it.
    private static string Initializer(string code, int open)
    {
        var depth = 0;
        for (var i = open; i < code.Length; i++)
        {
            if (code[i] == '{') depth++;
            else if (code[i] == '}' && --depth == 0)
                return code[open..(i + 1)];
        }
        return code[open..];
    }

    private static int LineOf(string code, int index) => code.Take(index).Count(c => c == '\n') + 1;

    [Fact]
    public void EveryProductLoaderReadsAnEmptyNumberAsMissing()
    {
        var sources = RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Select(f => (File: RepoSourceTree.RelativeToRoot(f), Text: File.ReadAllText(f))).ToList();
        // It read the tree: the layout authority is among what it scanned, with its initializers.
        Assert.Contains(sources, s => Path.GetFileName(s.File) == "ColumnLayoutInference.cs" && OptionsInitializer.Matches(s.Text).Count >= 4);
        var offenders = Offenders(sources).ToList();

        Assert.True(offenders.Count == 0,
            $"Set {Setting} on every TextLoader.Options (MLoop.Core.Data.MissingNumbers) and build loaders from options, " +
            "not LoadFromTextFile:" + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The scan can see: it flags an initializer without the setting and the overload, and passes one with it.</summary>
    [Fact]
    public void TheScanSeesALoaderLeftAtTheDefault()
    {
        var withoutIt = ("src/A.cs", """
            var options = new TextLoader.Options
            {
                Columns = [new TextLoader.Column("x", DataKind.Single, 0)],
                HasHeader = true,
            };
            """);
        var withIt = ("src/B.cs", """
            var options = new TextLoader.Options
            {
                Columns = [new TextLoader.Column("x", DataKind.Single, 0)],
                MissingRealsAsNaNs = true,
            };
            """);
        var overload = ("src/C.cs", """

            var data = ml.Data.LoadFromTextFile(path, columns, hasHeader: true);
            """);
        var comment = ("src/D.cs", "    // new TextLoader.Options { HasHeader = true } is what ML.NET's default gives");

        Assert.Equal(["src/A.cs:1", "src/C.cs:2"], Offenders([withoutIt, withIt, overload, comment]));
    }
}
