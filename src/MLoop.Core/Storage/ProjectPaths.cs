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
    /// segments, with the file system's own case rule, never as a string prefix.
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
}
