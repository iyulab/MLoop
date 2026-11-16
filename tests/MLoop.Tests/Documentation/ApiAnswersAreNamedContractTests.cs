using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// An HTTP answer of <c>mloop serve</c> is a named record, never an anonymous object. The record is where the field names
/// and kinds are written down: the compiler refuses an answer that is not one of them, the OpenAPI document describes each
/// instead of <c>object</c>, and the wire-shape baseline (<c>WireShapeContractTests</c>) can tell a renamed field from the
/// one it pinned. Before this, <c>GET /compare</c> could lose a field name and every API test still passed.
/// </summary>
public class ApiAnswersAreNamedContractTests
{
    // A call that writes an answer: the Results / TypedResults factories and the middleware's direct write.
    private static readonly Regex AnswerCall = new(@"\b(?:TypedResults|Results)\.\w+\(|\bWriteAsJsonAsync\(", RegexOptions.Compiled);

    // `new { … }` — an anonymous object; `new Name(…)` and `new Dictionary<…> { … }` carry a type and do not match.
    private static readonly Regex Anonymous = new(@"\bnew\s*\{", RegexOptions.Compiled);

    /// <summary>The text of the call that opens at <paramref name="open"/> (the index of its <c>(</c>), string literals skipped.</summary>
    private static string CallText(string source, int open)
    {
        var depth = 0;
        for (var i = open; i < source.Length; i++)
        {
            var c = source[i];
            if (c == '"')
            {
                // A string literal: its parentheses are text. A verbatim string's quotes double; an escaped one has a backslash.
                var verbatim = i > 0 && (source[i - 1] == '@' || (i > 1 && source[i - 1] == '$' && source[i - 2] == '@') || (i > 1 && source[i - 1] == '@' && source[i - 2] == '$'));
                for (i++; i < source.Length; i++)
                {
                    if (!verbatim && source[i] == '\\') { i++; continue; }
                    if (source[i] != '"') continue;
                    if (verbatim && i + 1 < source.Length && source[i + 1] == '"') { i++; continue; }
                    break;
                }
                continue;
            }
            if (c == '(') depth++;
            else if (c == ')' && --depth == 0)
                return source[open..(i + 1)];
        }
        return source[open..];
    }

    /// <summary>The calls in <paramref name="source"/> that write an answer, and the line each opens on.</summary>
    internal static IEnumerable<(int Line, string Text)> AnswerCalls(string source) =>
        AnswerCall.Matches(source).Select(m =>
            (source.Take(m.Index).Count(ch => ch == '\n') + 1, CallText(source, m.Index + m.Length - 1)));

    private static IEnumerable<string> Offenders(IEnumerable<(string File, string Source)> sources) =>
        from source in sources
        from call in AnswerCalls(source.Source)
        where Anonymous.IsMatch(call.Text)
        select $"{source.File}:{call.Line}";

    [Fact]
    public void EveryAnswerOfTheServerIsANamedRecord()
    {
        var sources = RepoSourceTree.ProductionSourceFiles("tools/MLoop.API")
            .Select(f => (File: RepoSourceTree.RelativeToRoot(f), Source: File.ReadAllText(f))).ToList();
        // It read the tree and saw the calls: the endpoint files are among what it scanned, with many answers between them.
        Assert.Contains(sources, s => Path.GetFileName(s.File) == "ModelEndpoints.cs");
        Assert.True(sources.Sum(s => AnswerCalls(s.Source).Count()) >= 60, "the scan found too few answer calls — its pattern stopped matching");

        var offenders = Offenders(sources).ToList();
        Assert.True(offenders.Count == 0,
            "Answer with a named record (see ExperimentResponses.cs and the other *Responses.cs files), not an anonymous object:" +
            Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The scan can see: every shape it forbids is found — one line, several lines, a second argument, an interpolated
    /// message that holds braces and parentheses — and the shapes it must let pass are not.</summary>
    [Fact]
    public void TheScanSeesAnAnonymousAnswer()
    {
        var forbidden = new[]
        {
            "return Results.Ok(new { a = 1 });",
            "return TypedResults.Ok(new\n{\n    a = 1,\n});",
            "return Results.Accepted($\"/jobs/{id}\", new { jobId = id });",
            "await context.Response.WriteAsJsonAsync(new { error = $\"{where} (a) {{x}} holds\" });",
            "return Results.Ok(rows.Select(r => new { r.A }));",
        };
        foreach (var text in forbidden)
            Assert.Single(Offenders([("X.cs", text)]));

        var allowed = new[]
        {
            "return Results.Ok(new Answer(1, \"x(\"));",
            "return Results.Ok(new Dictionary<string, object> { [\"a\"] = 1 });",
            "return TypedResults.BadRequest(new ErrorResponse($\"{where} (a)\"));",
        };
        foreach (var text in allowed)
            Assert.Empty(Offenders([("Y.cs", text)]));
    }
}
