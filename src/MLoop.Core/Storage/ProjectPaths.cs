namespace MLoop.Core.Storage;

/// <summary>
/// Where a path a user or a client gives names, and whether it stays in the project. A relative path is the
/// project's — its root, not the process's working directory — in every command and endpoint; outside a project
/// (a command that also works there) it is the working directory's. Each command used to restate the rule as its
/// own <c>IsPathRooted ? p : Combine(root, p)</c>, and the server checked containment with a prefix comparison,
/// which a sibling directory sharing the root's name (<c>/srv/proj-other</c> for <c>/srv/proj</c>) passed.
/// </summary>
public static class ProjectPaths
{
    /// <summary>
    /// <paramref name="path"/> as a full path: itself when rooted, otherwise under <paramref name="projectRoot"/>,
    /// or under the working directory when there is no project.
    /// </summary>
    public static string Resolve(string path, string? projectRoot) =>
        Path.GetFullPath(path, Path.GetFullPath(projectRoot ?? Directory.GetCurrentDirectory()));

    /// <summary>Whether two full paths name the same place, by the file system's own case rule.</summary>
    public static bool IsSame(string left, string right) =>
        Path.GetRelativePath(Path.GetFullPath(left), Path.GetFullPath(right)) == ".";

    /// <summary>
    /// Whether <paramref name="fullPath"/> is <paramref name="projectRoot"/> or lies under it — compared by path
    /// segments, with the file system's own case rule, never as a string prefix. Lexical: a symbolic link inside the
    /// project that points out of it still lies "under" the root here; where the path is about to be opened for a
    /// client, ask <see cref="IsWithinReal"/>.
    /// </summary>
    public static bool IsWithin(string fullPath, string projectRoot)
    {
        var relative = Path.GetRelativePath(Path.GetFullPath(projectRoot), Path.GetFullPath(fullPath));
        return relative == "."
               || !(Path.IsPathRooted(relative)
                    || relative == ".."
                    || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
                    || relative.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal));
    }

    /// <summary>
    /// Whether <paramref name="fullPath"/> lies under <paramref name="projectRoot"/> once every symbolic link (a file's
    /// or any directory's on the way) is followed. <see cref="IsWithin"/> compares the spelling of a path, so
    /// <c>datasets/link.csv</c> — a link in the project to a file outside it — passed it and the server then read the
    /// file the link named. A link that cannot be followed (a loop, no permission) counts as outside: refusing is the
    /// safe reading of a path whose end cannot be seen.
    /// </summary>
    public static bool IsWithinReal(string fullPath, string projectRoot)
    {
        try
        {
            return IsWithin(RealPath(fullPath), RealPath(projectRoot));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// <paramref name="path"/> as a full path with every symbolic link on it replaced by what it points to. The part of
    /// the path that does not exist yet (a file about to be created) is kept as written.
    /// </summary>
    /// <exception cref="IOException">Links that lead back to themselves, or chain deeper than any real layout.</exception>
    public static string RealPath(string path) => RealPath(Path.GetFullPath(path), 0);

    private const int MaxLinkDepth = 40;

    private static string RealPath(string fullPath, int depth)
    {
        if (depth > MaxLinkDepth)
            throw new IOException($"Too many symbolic links while resolving '{fullPath}'.");

        var root = Path.GetPathRoot(fullPath) ?? "";
        var current = root;
        var parts = fullPath[root.Length..].Split(
            [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < parts.Length; i++)
        {
            current = Path.Combine(current, parts[i]);
            FileSystemInfo? entry = Directory.Exists(current) ? new DirectoryInfo(current)
                : File.Exists(current) ? new FileInfo(current)
                : null;
            if (entry?.LinkTarget is null)
                continue;

            // The link's target may itself sit behind other links, so it is resolved in turn; the rest of the path follows it.
            var target = Path.GetFullPath(entry.LinkTarget, Path.GetDirectoryName(current) ?? root);
            var rest = string.Join(Path.DirectorySeparatorChar, parts[(i + 1)..]);
            return RealPath(rest.Length == 0 ? target : Path.Combine(target, rest), depth + 1);
        }
        return current;
    }
}
