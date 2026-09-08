using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A trainer substitution is stated twice — once as prose for a person
/// (<c>TrainerDescriptor.FallbackReason</c>) and once as a value for code
/// (<c>TrainerDescriptor.FallbackKind</c>). Wherever one is set, the other is set with it.
/// </summary>
/// <remarks>
/// <para>
/// The pair exists because the two audiences need different things: the reason carries the
/// specifics a person wants (<c>Accuracy→F1Score (AUC imbalance)</c>), while the decision the
/// engine makes — whether a second AutoML pass could possibly do better — must not be recovered by
/// matching that text's prefix, because then a reworded message silently changes behavior.
/// </para>
/// <para>
/// Two fields naming one fact is a drift risk taken deliberately, and this is the guard that pays
/// for it. A new fallback path that sets only the reason leaves the kind at its default
/// <c>None</c>, which reads as "this run did what it meant to" — the engine would then run a second
/// search that repeats the first one's failure, which is precisely the defect the kind was added to
/// fix.
/// </para>
/// </remarks>
public class FallbackKindPairingContractTests
{
    /// <summary>Where a <c>TrainerDescriptor</c> object initializer begins.</summary>
    /// <remarks>
    /// The body is then taken by counting braces rather than by a regex. The first version of this
    /// guard matched a body of "anything but braces" and saw one of the two fallback sites: the
    /// other sets its reason from an interpolated string, whose own <c>{}</c> ended the match early.
    /// It reported "1 site visible" on its first run, which is the only reason it is not still
    /// silently checking half of what it claims to.
    /// </remarks>
    private static readonly Regex InitializerStart =
        new(@"new\s+TrainerDescriptor\s*\{", RegexOptions.Compiled);

    private static IEnumerable<string> InitializerBodies(string source)
    {
        foreach (var start in InitializerStart.Matches(source).Cast<Match>())
        {
            var open = start.Index + start.Length - 1;
            var depth = 0;

            for (var i = open; i < source.Length; i++)
            {
                if (source[i] == '{') depth++;
                else if (source[i] == '}' && --depth == 0)
                {
                    yield return source[(open + 1)..i];
                    break;
                }
            }
        }
    }

    private static readonly Regex SetsReason = new(@"\bFallbackReason\s*=", RegexOptions.Compiled);
    private static readonly Regex SetsKind = new(@"\bFallbackKind\s*=", RegexOptions.Compiled);

    [Fact]
    public void EveryFallbackReasonIsSetAlongsideItsKind()
    {
        var offenders =
            (from file in RepoSourceTree.ProductionSourceFiles("src", "tools")
             where !Path.GetFileName(file).Equals("TrainerDescriptor.cs", StringComparison.Ordinal)
             let text = File.ReadAllText(file)
             from body in InitializerBodies(text)
             where SetsReason.IsMatch(body) != SetsKind.IsMatch(body)
             select $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, file)} — a TrainerDescriptor "
                  + $"sets {(SetsReason.IsMatch(body) ? "FallbackReason without FallbackKind" : "FallbackKind without FallbackReason")}")
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            "A trainer substitution states itself to a person and to code, or to neither:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>
    /// The guard is only worth having if it can see the initializers it is meant to police. Two
    /// exist today (the metric substitution and the direct-pipeline stand-in); a rewrite that moved
    /// them out of object-initializer form would leave this test green while checking nothing.
    /// </summary>
    [Fact]
    public void TheGuardSeesTheFallbackSitesItPolices()
    {
        var initializersSettingAFallback =
            (from file in RepoSourceTree.ProductionSourceFiles("src", "tools")
             let text = File.ReadAllText(file)
             from body in InitializerBodies(text)
             where SetsReason.IsMatch(body)
             select body).Count();

        Assert.True(
            initializersSettingAFallback >= 2,
            $"Expected the two known fallback sites to be visible to this scan; saw "
            + $"{initializersSettingAFallback}. If the sites moved, move the guard with them — a "
            + "scan that matches nothing passes for the wrong reason.");
    }
}
