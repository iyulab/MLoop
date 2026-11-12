using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Common;

/// <summary>The shape of a JSON answer — path → kinds — and its baseline text form.</summary>
public static class WireShape
{
    public static SortedDictionary<string, string> Describe(string json)
    {
        using var document = JsonDocument.Parse(json);
        var kinds = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        Walk(document.RootElement, "$", kinds);
        return new SortedDictionary<string, string>(
            kinds.ToDictionary(pair => pair.Key, pair => string.Join('|', pair.Value)), StringComparer.Ordinal);
    }

    private static void Walk(JsonElement element, string path, SortedDictionary<string, SortedSet<string>> kinds)
    {
        void Add(string kind)
        {
            if (!kinds.TryGetValue(path, out var set)) kinds[path] = set = new SortedSet<string>(StringComparer.Ordinal);
            set.Add(kind);
        }

        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                Add("object");
                foreach (var property in element.EnumerateObject()) Walk(property.Value, path + "." + property.Name, kinds);
                break;
            case JsonValueKind.Array:
                Add("array");
                foreach (var item in element.EnumerateArray()) Walk(item, path + "[]", kinds);
                break;
            case JsonValueKind.String: Add("string"); break;
            case JsonValueKind.Number: Add("number"); break;
            case JsonValueKind.True:
            case JsonValueKind.False: Add("boolean"); break;
            default: Add("null"); break;
        }
    }

    /// <summary>One block per scenario: <c>## id | header</c>, then <c>path: kinds</c> lines.</summary>
    public static string Render(IEnumerable<(string Id, string Header, IReadOnlyDictionary<string, string> Shape)> blocks)
    {
        var text = new StringBuilder();
        foreach (var (id, header, shape) in blocks)
        {
            text.Append("## ").Append(id).Append(" | ").Append(header).Append('\n');
            foreach (var (path, kinds) in shape.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                text.Append(path).Append(": ").Append(kinds).Append('\n');
        }
        return text.ToString();
    }

    public static Dictionary<string, HashSet<string>> Parse(string text)
    {
        var blocks = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        HashSet<string>? current = null;
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0) continue;
            if (line.StartsWith("## ", StringComparison.Ordinal))
            {
                var id = line[3..].Split(" | ")[0];
                current = blocks[id] = [line];
            }
            else current?.Add(line);
        }
        return blocks;
    }

    /// <summary>
    /// What differs between the baseline and what was observed: per scenario a changed status, and per path one that is gone
    /// (<c>removed</c>) or new (<c>added</c>) - a path whose kinds changed is both. <paramref name="ignoreNull"/> leaves out of the
    /// comparison the <c>null</c> kind and any path that was only ever null: an output that writes an absent value as null (or
    /// omits it) says nothing of the kind the value has, and a scenario that cannot fix whether the value is there must not fail on it.
    /// </summary>
    public static List<string> Differences(Dictionary<string, HashSet<string>> baseline, Dictionary<string, HashSet<string>> observed,
        bool ignoreNull = false)
    {
        var differences = new List<string>();
        foreach (var id in baseline.Keys.Union(observed.Keys).OrderBy(k => k, StringComparer.Ordinal))
        {
            if (!baseline.TryGetValue(id, out var was)) { differences.Add($"{id}: scenario is not in the baseline"); continue; }
            if (!observed.TryGetValue(id, out var now)) { differences.Add($"{id}: scenario was not observed"); continue; }

            var wasHeader = was.Single(l => l.StartsWith("## ", StringComparison.Ordinal));
            var nowHeader = now.Single(l => l.StartsWith("## ", StringComparison.Ordinal));
            if (wasHeader != nowHeader) differences.Add($"{id}: status changed (was {wasHeader}, now {nowHeader})");

            var wasPaths = Paths(was, ignoreNull);
            var nowPaths = Paths(now, ignoreNull);
            foreach (var path in wasPaths.Keys.OrderBy(k => k, StringComparer.Ordinal))
                if (!nowPaths.TryGetValue(path, out var kinds) || !SameKinds(kinds, wasPaths[path])) differences.Add($"{id}: removed {path}: {wasPaths[path]}");
            foreach (var path in nowPaths.Keys.OrderBy(k => k, StringComparer.Ordinal))
                if (!wasPaths.TryGetValue(path, out var kinds) || !SameKinds(kinds, nowPaths[path])) differences.Add($"{id}: added {path}: {nowPaths[path]}");
        }
        return differences;
    }

    private static Dictionary<string, string> Paths(HashSet<string> block, bool ignoreNull)
    {
        var paths = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var line in block.Where(l => !l.StartsWith("## ", StringComparison.Ordinal)))
        {
            var split = line.IndexOf(": ", StringComparison.Ordinal);
            var path = line[..split];
            var kinds = line[(split + 2)..];
            if (ignoreNull)
            {
                kinds = string.Join('|', kinds.Split('|').Where(k => k != "null"));
                if (kinds.Length == 0) kinds = Unseen;
            }
            paths[path] = kinds;
        }
        return paths;
    }

    /// <summary>The kind of a path that was only ever null where nulls are tolerated.</summary>
    private const string Unseen = "*";

    private static bool SameKinds(string a, string b) => a == b || a == Unseen || b == Unseen;

    public static List<string> Unclaimed(IEnumerable<string> mapped, ISet<string> claimed) =>
        mapped.Where(route => !claimed.Contains(route)).Distinct().OrderBy(r => r, StringComparer.Ordinal).ToList();
}
