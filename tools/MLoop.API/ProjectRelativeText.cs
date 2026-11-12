namespace MLoop.API;

/// <summary>
/// What a response says about the server's disk, in the project's terms. A job records the absolute paths it trains with and a
/// failure's message quotes them (<c>No class subfolders found in '/srv/…/datasets/x'</c>); a caller of a training service
/// that lives on another host was handed that host's directory layout. The project's own paths are the ones the request
/// used — relative to the project — so the answer speaks in them. Only the project root is removed: a path outside it
/// (a system file an exception names) is not this layer's to hide or to rewrite.
/// </summary>
public static class ProjectRelativeText
{
    /// <summary>The text with every occurrence of the project root taken off the front of the paths under it (the root itself becomes <c>.</c>).</summary>
    public static string? In(string? text, string projectRoot)
    {
        if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(projectRoot))
            return text;
        var root = projectRoot.TrimEnd('/', '\\');
        if (root.Length == 0)
            return text;
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

        // The root is written with the separators of the host that wrote it and quoted with either: take the prefix off in both forms.
        foreach (var form in new[] { root, root.Replace('\\', '/'), root.Replace('/', '\\') }.Distinct(StringComparer.Ordinal))
        {
            text = text.Replace(form + "/", "", comparison).Replace(form + "\\", "", comparison);
            text = ReplaceWholeRoot(text, form, comparison);
        }
        return text;
    }

    public static IReadOnlyList<string>? In(IReadOnlyList<string>? texts, string projectRoot)
        => texts?.Select(t => In(t, projectRoot)!).ToList();

    // A root that is not followed by more path (an end, a quote, a space) stands for the project itself.
    private static string ReplaceWholeRoot(string text, string root, StringComparison comparison)
    {
        var at = 0;
        while ((at = text.IndexOf(root, at, comparison)) >= 0)
        {
            var end = at + root.Length;
            var continues = end < text.Length && (char.IsLetterOrDigit(text[end]) || text[end] is '_' or '-' or '.');
            if (continues) { at = end; continue; }
            text = string.Concat(text.AsSpan(0, at), ".", text.AsSpan(end));
            at += 1;
        }
        return text;
    }
}
