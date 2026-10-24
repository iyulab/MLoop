using System.Text.RegularExpressions;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Changes to <c>mloop.yaml</c> made on its lines rather than by writing the parsed configuration back:
/// a re-serialized file loses every comment (<c>mloop init</c> writes fifteen, the guide to
/// <c>time_limit_seconds</c> among them) and the order the user gave the keys. Each edit returns
/// <see langword="null"/> when the text is not in the block shape it knows — the caller then writes the
/// file whole, as before.
/// </summary>
public static partial class MLoopYamlEdit
{
    /// <summary>
    /// <paramref name="yaml"/> with <c>models.&lt;model&gt;.&lt;key&gt;</c> set to each value: an existing
    /// line keeps its indentation and trailing comment, a missing key is added as the block's first line.
    /// </summary>
    public static string? SetModelScalars(string yaml, string model, IReadOnlyList<(string Key, string Value)> values)
    {
        var (lines, newline) = Split(yaml);
        if (FindModel(lines, model) is not { } found)
            return null;

        foreach (var (key, value) in values)
        {
            var existing = -1;
            for (var i = found.Start + 1; i < found.End; i++)
            {
                if (Indent(lines[i]) == found.ChildIndent && KeyOf(lines[i]) == key)
                {
                    existing = i;
                    break;
                }
            }

            if (existing < 0)
            {
                lines.Insert(found.Start + 1, new string(' ', found.ChildIndent) + key + ": " + Scalar(value));
                found = found with { End = found.End + 1 };
                continue;
            }

            var match = ScalarLine().Match(lines[existing]);
            // A key whose value is a nested block or a flow collection is not a scalar to replace.
            if (!match.Success || match.Groups["value"].Value.Trim() is "" or ['[' or '{' or '|' or '>', ..])
                return null;
            lines[existing] = match.Groups["head"].Value + Scalar(value) + match.Groups["comment"].Value;
        }
        return string.Join(newline, lines);
    }

    /// <summary>
    /// <paramref name="yaml"/> with <paramref name="block"/> — a model's definition as YAML, unindented, its
    /// first line <c>&lt;model&gt;:</c> — appended to the <c>models:</c> mapping.
    /// </summary>
    public static string? AddModel(string yaml, string block)
    {
        var (lines, newline) = Split(yaml);
        if (FindModels(lines) is not { } models)
            return null;
        var indent = new string(' ', models.ChildIndent);
        var added = Split(block.TrimEnd()).Lines.Select(l => l.Length == 0 ? l : indent + l);
        lines.InsertRange(models.End, added);
        return string.Join(newline, lines);
    }

    private sealed record Block(int Start, int End, int ChildIndent);

    // The `models:` mapping: its line, the line after its last child, the children's indentation.
    private static Block? FindModels(List<string> lines)
    {
        var start = lines.FindIndex(l => Indent(l) == 0 && KeyOf(l) == "models" && ScalarLine().Match(l).Groups["value"].Value.Trim() == "");
        return start < 0 ? null : BlockAt(lines, start, 0);
    }

    private static Block? FindModel(List<string> lines, string model)
    {
        if (FindModels(lines) is not { } models)
            return null;
        for (var i = models.Start + 1; i < models.End; i++)
        {
            if (Indent(lines[i]) == models.ChildIndent && KeyOf(lines[i]) == model)
                return ScalarLine().Match(lines[i]).Groups["value"].Value.Trim() == "" ? BlockAt(lines, i, models.ChildIndent) : null;
        }
        return null;
    }

    // The lines under `lines[start]` (indented deeper than `indent`, with blank and comment lines between them).
    private static Block? BlockAt(List<string> lines, int start, int indent)
    {
        var end = start + 1;
        var childIndent = -1;
        for (var i = start + 1; i < lines.Count; i++)
        {
            if (IsBlankOrComment(lines[i]))
                continue;
            if (Indent(lines[i]) <= indent)
                break;
            if (childIndent < 0)
                childIndent = Indent(lines[i]);
            end = i + 1;
        }
        return childIndent < 0 ? null : new Block(start, end, childIndent);
    }

    private static (List<string> Lines, string Newline) Split(string text)
    {
        var newline = text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
        return ([.. text.Split(newline)], newline);
    }

    private static bool IsBlankOrComment(string line) => line.TrimStart() is "" or ['#', ..];

    private static int Indent(string line) => line.Length - line.TrimStart(' ').Length;

    private static string? KeyOf(string line)
    {
        var match = ScalarLine().Match(line);
        return match.Success ? match.Groups["key"].Value : null;
    }

    // A value as a plain scalar when it is one; otherwise double-quoted.
    private static string Scalar(string value) => PlainScalar().IsMatch(value)
        ? value
        : "\"" + value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal) + "\"";

    [GeneratedRegex(@"^(?<head>\s*(?<key>[A-Za-z0-9_\-]+):[ \t]*)(?<value>[^#]*?)(?<comment>[ \t]+#.*)?$")]
    private static partial Regex ScalarLine();

    [GeneratedRegex(@"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")]
    private static partial Regex PlainScalar();
}
