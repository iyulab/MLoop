using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Product code reads a number from text with a stated culture — invariant, as ML.NET's loader reads a file — never
/// the machine's. A parse that takes the current culture reads "1.5" as a number on one machine and not on another
/// (a comma-decimal culture): on such a machine regression refused every label as "no valid numeric values", and the
/// saved schema recorded numeric columns as categories.
/// </summary>
public class NumberParsingCultureContractTests
{
    private static readonly Regex NumberParse = new(@"\b(?:double|float|decimal|int|long|short|byte)\.(?:Try)?Parse\(", RegexOptions.Compiled);

    private static IEnumerable<string> Offenders(IEnumerable<(string File, string Text)> sources)
    {
        foreach (var (file, text) in sources)
        {
            var code = string.Join('\n', text.Split('\n').Select(l => l.TrimStart().StartsWith("//", StringComparison.Ordinal) ? "" : l));
            foreach (Match m in NumberParse.Matches(code))
            {
                var call = Call(code, m.Index + m.Length - 1);
                if (!call.Contains("CultureInfo", StringComparison.Ordinal) && !call.Contains("NumberFormatInfo", StringComparison.Ordinal))
                    yield return $"{file}:{code.Take(m.Index).Count(c => c == '\n') + 1}";
            }
        }
    }

    // The argument list from its opening parenthesis to the one that closes it.
    private static string Call(string code, int open)
    {
        var depth = 0;
        for (var i = open; i < code.Length; i++)
        {
            if (code[i] == '(') depth++;
            else if (code[i] == ')' && --depth == 0)
                return code[open..(i + 1)];
        }
        return code[open..];
    }

    [Fact]
    public void ProductCodeParsesNumbersWithAStatedCulture()
    {
        var sources = RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Select(f => (File: RepoSourceTree.RelativeToRoot(f), Text: File.ReadAllText(f))).ToList();
        // It read the tree: a known invariant parse is among what it scanned.
        Assert.Contains(sources, s => Path.GetFileName(s.File) == "FeedbackValueComparer.cs" && NumberParse.IsMatch(s.Text));
        var offenders = Offenders(sources).ToList();

        Assert.True(offenders.Count == 0,
            "Parse a number from text with CultureInfo.InvariantCulture (as the loader reads it), not the machine's culture:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The scan can see: it flags a parse without a culture, and passes one with it across lines and a comment.</summary>
    [Fact]
    public void TheScanSeesAParseThatTakesTheMachinesCulture()
    {
        var bare = ("src/A.cs", "        if (double.TryParse(value, out var v)) return v;");
        var stated = ("src/B.cs", """
                    if (double.TryParse(value, NumberStyles.Float,
                            CultureInfo.InvariantCulture, out var v))
            """);
        var nested = ("src/C.cs", "        var n = int.Parse(Trim(s), System.Globalization.CultureInfo.InvariantCulture);");
        var comment = ("src/D.cs", "        // double.TryParse(value, out _) reads the machine's culture");

        Assert.Equal(["src/A.cs:1"], Offenders([bare, stated, nested, comment]));
    }
}
