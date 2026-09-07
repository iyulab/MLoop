namespace MLoop.API;

/// <summary>
/// The project this server serves, resolved once.
/// </summary>
/// <param name="Value">The absolute project root.</param>
/// <remarks>
/// Ten places used to answer this question for themselves — six service registrations and three
/// endpoints, each calling <c>FindRoot()</c>, plus the startup log. That is ten walks up the
/// filesystem for one fact that cannot change while the process runs, and ten places for the answer
/// to drift if one of them ever gained a condition. Registering the answer instead of the question
/// makes the resolution a startup event: it happens once, before the first request, and a
/// deployment that cannot find its project fails while an operator is still watching the log rather
/// than as a 500 from whichever endpoint got there first.
/// </remarks>
public sealed record ProjectRootPath(string Value)
{
    /// <summary>The root as a plain path, so a consumer reads it like the string it replaced.</summary>
    public static implicit operator string(ProjectRootPath root) => root.Value;
}
