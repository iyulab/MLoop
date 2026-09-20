using System.Text.RegularExpressions;

namespace MLoop.Core.Storage;

/// <summary>
/// Single source of truth for what a model name may be. A model name becomes a directory under
/// <see cref="ExperimentLayout.ModelsDirectory"/>, so the rule is a filesystem contract, not a
/// presentation preference — which is why it lives beside the layout authority in
/// <c>MLoop.Core</c> rather than in the CLI: <c>MLoop.Ops</c> and <c>MLoop.API</c> resolve the same
/// directories and cannot reference <c>MLoop.CLI</c>.
/// </summary>
/// <remarks>
/// <para>This knowledge existed in three places and two of them agreed. <c>InitCommand</c> and
/// <c>ModelNameResolver</c> both required 2–50 characters, lowercase, hyphen-separated, no reserved
/// word. <c>ValidateCommand</c> required none of that: it accepted uppercase, underscores, a
/// leading underscore, any length, and every reserved word — while printing "must be lowercase
/// alphanumeric with hyphens only", a sentence describing the rule it did not implement.</para>
/// <para>The gap was reachable, and it defeated the one command meant to close it: a
/// <c>mloop.yaml</c> naming a model <c>MyModel_v2</c> passed <c>mloop validate</c> clean, then
/// failed at <c>mloop train</c> when the resolver rejected the same name. Validation exists to find
/// that before the work starts.</para>
/// <para>The strict rule is the one kept. A model name is a directory name, so permissiveness here
/// is not generosity — it defers the rejection to the filesystem, on a platform-dependent schedule,
/// after the user has spent a training run.</para>
/// </remarks>
public static partial class ModelName
{
    /// <summary>
    /// Names that cannot be used as model names because the layout already gives them a meaning
    /// inside a model directory (or beside it). Exposed so a message naming them cannot drift from
    /// the check — the prose copy in <c>init</c>'s error text used to be a fourth place to edit.
    /// </summary>
    public static readonly IReadOnlyList<string> ReservedNames =
    [
        ExperimentLayout.StagingDirectory,
        ExperimentLayout.ProductionDirectory,
        "temp",
        "cache",
        "index",
        "registry"
    ];

    /// <summary>Shortest accepted name. One character is too easy to typo into a real name.</summary>
    public const int MinLength = 2;

    /// <summary>Longest accepted name, chosen to stay clear of path-length limits once the layout's
    /// own segments (<c>models/</c>, <c>staging/exp-NNN/</c>, a file name) are appended.</summary>
    public const int MaxLength = 50;

    /// <summary>
    /// True when <paramref name="name"/> is usable as a model name as given. Does not normalize:
    /// callers that accept user input normalize first with <see cref="Normalize"/>, and callers
    /// validating a stored value check what is actually stored.
    /// </summary>
    public static bool IsValid(string? name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        if (name.Length < MinLength || name.Length > MaxLength)
            return false;

        foreach (var reserved in ReservedNames)
        {
            if (string.Equals(name, reserved, StringComparison.OrdinalIgnoreCase))
                return false;
        }

        return NamePattern().IsMatch(name);
    }

    /// <summary>
    /// The name as it is written on disk: trimmed and lowercased. Normalizing does not make a name
    /// valid — <c>My_Model</c> normalizes to <c>my_model</c>, which <see cref="IsValid"/> still
    /// rejects — so a caller taking user input normalizes and then validates.
    /// </summary>
    public static string Normalize(string name) => name.Trim().ToLowerInvariant();

    /// <summary>
    /// One sentence naming the rule <paramref name="name"/> breaks, or <c>null</c> when it breaks
    /// none. A single generic sentence cannot say why <c>staging</c> was refused, and a user told
    /// "lowercase alphanumeric with hyphens" about a name that already is one reads it as a bug.
    /// </summary>
    public static string? DescribeViolation(string? name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return "A model name cannot be empty.";

        if (name.Length < MinLength)
            return $"A model name needs at least {MinLength} characters.";

        if (name.Length > MaxLength)
            return $"A model name can be at most {MaxLength} characters; this one is {name.Length}.";

        foreach (var reserved in ReservedNames)
        {
            if (string.Equals(name, reserved, StringComparison.OrdinalIgnoreCase))
                return $"'{reserved}' is reserved by the project layout. Reserved names: {string.Join(", ", ReservedNames)}.";
        }

        return NamePattern().IsMatch(name)
            ? null
            : "A model name is lowercase letters, digits and single hyphens, starting with a letter (e.g. my-model).";
    }

    [GeneratedRegex(@"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")]
    private static partial Regex NamePattern();
}
