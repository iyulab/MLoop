using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>What a column of an inferred layout is for.</summary>
public enum ColumnLayoutPurpose
{
    /// <summary>Present in the file but not a feature.</summary>
    Ignore,

    /// <summary>The column a model learns to predict.</summary>
    Label,

    /// <summary>A number or a boolean, used as it is.</summary>
    Numeric,

    /// <summary>A string holding categories.</summary>
    Categorical,

    /// <summary>A string holding language.</summary>
    Text,
}

/// <summary>
/// One column of an inferred layout: a column of <see cref="ColumnLayout.Options"/> (which may cover
/// several file columns) and what it is for.
/// </summary>
public sealed record ColumnLayoutColumn(string Name, ColumnLayoutPurpose Purpose);

/// <summary>
/// One column of the file, as the layout read it before folding adjacent columns into vectors: its header name, the
/// kind it loads as, and its purpose.
/// </summary>
public sealed record ColumnLayoutFileColumn(string Name, DataKind Kind, ColumnLayoutPurpose Purpose);

/// <summary>
/// How a delimited file is read: the loader options (separator, quoting, and the columns — adjacent
/// columns of one kind folded into one vector) and the purpose of each loaded column.
/// </summary>
/// <param name="Options">The loader options.</param>
/// <param name="Label">The label column, when one was named.</param>
/// <param name="Columns">The loaded columns — a vector of folded columns is one entry, named for the vector.</param>
/// <param name="FileColumns">
/// Every column of the file by its own name — what a column inside a folded vector was read as, which
/// <paramref name="Columns"/> says only of the vector.
/// </param>
public sealed record ColumnLayout(
    TextLoader.Options Options, string? Label, IReadOnlyList<ColumnLayoutColumn> Columns, IReadOnlyList<ColumnLayoutFileColumn> FileColumns)
{
    /// <summary>The loaded columns with <paramref name="purpose"/>, in load order.</summary>
    public IReadOnlyList<string> Of(ColumnLayoutPurpose purpose) =>
        [.. Columns.Where(c => c.Purpose == purpose).Select(c => c.Name)];

    /// <summary>The file column named <paramref name="name"/>, or <see langword="null"/>.</summary>
    public ColumnLayoutFileColumn? FileColumn(string name) => FileColumns.FirstOrDefault(c => c.Name == name);
}

/// <summary>A file whose layout cannot be inferred — it does not split into consistent columns, or has no rows.</summary>
public sealed class ColumnLayoutException(string message) : Exception(message);

/// <summary>
/// Infers the column layout of a delimited file with a header — the layout every saved model's input
/// contract rests on, so it reproduces the layout ML.NET AutoML's <c>InferColumns</c> gives the same
/// file, field for field.
/// </summary>
/// <remarks>
/// <para>
/// A model reads its input through the <see cref="TextLoader.Options"/> it was trained with, and the
/// numeric columns reach it as one vector whose width is fixed at training. Inferring a different
/// layout for the same file — one more column read as text, one range split differently — changes that
/// width, and a saved model can no longer read its input. So the rules here are the established ones,
/// kept as they are:
/// </para>
/// <list type="bullet">
/// <item>The split: the first of (sparse no/yes) × (quoting yes/no) × (multi-line no/yes) under which
/// at least 98% of the first 1,000 lines have the same field count, greater than one.</item>
/// <item>Types, from the first 10,000 lines of the sample, the header line included: a column every
/// value of which parses as a float (an empty value, <c>NaN</c>, <c>?</c> and <c>NA</c> included) is
/// <see cref="DataKind.Single"/>; else one whose every value is a non-empty boolean is
/// <see cref="DataKind.Boolean"/>; else <see cref="DataKind.String"/>. The label is a boolean whenever
/// its values all read as one — so a <c>0</c>/<c>1</c> label is.</item>
/// <item>Purposes, from the first 1,000 rows: a string column with fewer than 70% distinct values is
/// categorical; one with at least 85% that reads as language (long, or with spaces) is text; anything
/// else is ignored. Numbers and booleans are numeric.</item>
/// <item>Grouping: the numeric columns of one kind fold into one vector (<c>Features</c>, or
/// <c>BooleanFeatures</c>) over ascending, merged ranges; every other column stays its own, under its
/// header name.</item>
/// </list>
/// <para>
/// Two things differ, deliberately. A file over 8 MB is sampled from fixed positions (the same file
/// always gives the same layout, where the original draws them at random). And no column is given an
/// image-path purpose — MLoop does not read image paths from a table.
/// </para>
/// </remarks>
public static class ColumnLayoutInference
{
    private const int SplitRows = 1_000;
    private const double UniformColumnCountShare = 0.98;
    private const int TypeRows = 10_000;
    private const int PurposeRows = 1_000;
    private const int MaxColumns = 10_000;

    /// <summary>
    /// The loaded name of the feature columns of one kind that fold together — the one column's own name, or
    /// the vector's (<c>Features</c> for numbers, <c>BooleanFeatures</c> for yes/no columns). A reader that
    /// rebuilds the loaded view from named columns (serving) names its vectors by this, as the loader does.
    /// </summary>
    public static string FoldedName(IReadOnlyList<string> members, DataKind kind)
    {
        ArgumentNullException.ThrowIfNull(members);
        return members.Count == 1 ? members[0] : GroupName(ColumnLayoutPurpose.Numeric, kind);
    }

    /// <summary>
    /// Infers the layout of <paramref name="path"/>, a file with a header whose column <paramref name="label"/> is the
    /// label — or, when <paramref name="label"/> is null, a file without one (clustering, anomaly detection): every
    /// column is then typed and grouped as a feature, so no column is set apart from the numeric vector.
    /// </summary>
    /// <exception cref="ColumnLayoutException">The file does not split into consistent columns, or has no data rows.</exception>
    /// <exception cref="ArgumentException">No column is named <paramref name="label"/>.</exception>
    public static ColumnLayout Infer(MLContext mlContext, string path, string? label, char separator = ',')
    {
        var sample = ColumnLayoutSample.Read(path);
        var split = InferSplit(mlContext, sample, separator);
        var columns = InferTypes(mlContext, sample, split, label);

        var typed = new TextLoader.Options
        {
            Columns = [.. columns.Select(c => new TextLoader.Column(c.Name, c.Kind, c.Index))],
            Separators = [separator],
            AllowSparse = split.Sparse,
            AllowQuoting = split.Quoting,
            ReadMultilines = split.Multiline,
            HasHeader = true,
            MissingRealsAsNaNs = true,
        };
        var purposes = InferPurposes(mlContext, mlContext.Data.CreateTextLoader(typed).Load(path), columns, label);

        var groups = GroupColumns(columns, purposes);
        var options = new TextLoader.Options
        {
            Columns = [.. groups.Select(g => new TextLoader.Column(g.Name, g.Kind, g.Ranges))],
            AllowQuoting = split.Quoting,
            AllowSparse = split.Sparse,
            Separators = [separator],
            ReadMultilines = split.Multiline,
            HasHeader = true,
            // An empty number is missing, not zero (MissingNumbers).
            MissingRealsAsNaNs = true,
        };
        return new ColumnLayout(options, label, [.. groups.Select(g => new ColumnLayoutColumn(g.Name, g.Purpose))],
            [.. columns.Select((c, i) => new ColumnLayoutFileColumn(c.Name, c.Kind, purposes[i]))]);
    }

    private sealed record Split(char Separator, bool Sparse, bool Quoting, bool Multiline, int ColumnCount);

    private sealed record TypedColumn(int Index, string Name, DataKind Kind);

    private sealed record Group(string Name, DataKind Kind, ColumnLayoutPurpose Purpose, TextLoader.Range[] Ranges);

    private static Split InferSplit(MLContext mlContext, IMultiStreamSource sample, char separator)
    {
        foreach (var sparse in new[] { false, true })
            foreach (var quoting in new[] { true, false })
                foreach (var multiline in new[] { false, true })
                {
                    var options = new TextLoader.Options
                    {
                        Columns = [new TextLoader.Column("C", DataKind.String, [new TextLoader.Range(0, null)])],
                        Separators = [separator],
                        AllowQuoting = quoting,
                        AllowSparse = sparse,
                        ReadMultilines = multiline,
                        MissingRealsAsNaNs = true,
                    };
                    if (FieldCount(mlContext, options, sample) is { } count)
                        return new Split(separator, sparse, quoting, multiline, count);
                }

        throw new ColumnLayoutException(
            "Unable to split the file provided into multiple, consistent columns. " +
            "Readable formats include delimited files such as CSV/TSV. " +
            "Check for a consistent number of columns and proper escaping and quoting.");
    }

    // The field count nearly every line has, or null when the lines disagree or have one field.
    private static int? FieldCount(MLContext mlContext, TextLoader.Options options, IMultiStreamSource sample)
    {
        try
        {
            var data = mlContext.Data.TakeRows(mlContext.Data.CreateTextLoader(options, sample).Load(sample), SplitRows);
            var counts = new List<int>();
            var column = data.Schema["C"];
            using (var cursor = data.GetRowCursor([column]))
            {
                var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(column);
                VBuffer<ReadOnlyMemory<char>> line = default;
                while (cursor.MoveNext())
                {
                    getter(ref line);
                    counts.Add(line.Length);
                }
            }

            var common = counts.GroupBy(c => c).OrderByDescending(g => g.Count()).First();
            return common.Count() >= UniformColumnCountShare * counts.Count && common.Key > 1 ? common.Key : null;
        }
        catch (Exception)
        {
            // A combination the loader cannot read with is simply not the answer.
            return null;
        }
    }

    private static TypedColumn[] InferTypes(MLContext mlContext, IMultiStreamSource sample, Split split, string? label)
    {
        if (split.ColumnCount >= MaxColumns)
            throw new ColumnLayoutException("Unable to infer column types of the file provided.");

        var options = new TextLoader.Options
        {
            Columns = [new TextLoader.Column("C", DataKind.String, 0, split.ColumnCount - 1)],
            Separators = [split.Separator],
            AllowSparse = split.Sparse,
            AllowQuoting = split.Quoting,
            ReadMultilines = split.Multiline,
            MissingRealsAsNaNs = true,
        };
        var data = mlContext.Data.TakeRows(mlContext.Data.CreateTextLoader(options).Load(sample), TypeRows);

        var rows = new List<ReadOnlyMemory<char>[]>();
        using (var cursor = data.GetRowCursor(data.Schema))
        {
            var column = cursor.Schema["C"];
            if (column.Type is VectorDataViewType)
            {
                var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(column);
                VBuffer<ReadOnlyMemory<char>> line = default;
                while (cursor.MoveNext())
                {
                    getter(ref line);
                    var values = new ReadOnlyMemory<char>[split.ColumnCount];
                    line.CopyTo(values);
                    rows.Add(values);
                }
            }
            else
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(column);
                ReadOnlyMemory<char> value = default;
                while (cursor.MoveNext())
                {
                    getter(ref value);
                    rows.Add([value]);
                }
            }
        }

        // The header line and at least one row.
        if (rows.Count < 2)
            throw new ColumnLayoutException("Unable to infer column types of the file provided.");

        var used = new HashSet<string>();
        var columns = new TypedColumn[split.ColumnCount];
        for (var i = 0; i < split.ColumnCount; i++)
        {
            var values = rows.Select(r => r[i]).ToArray();
            var kind = values.Skip(1).All(v => ColumnLayoutValues.IsSingle(v.Span)) ? DataKind.Single
                : AllBoolean(values) ? DataKind.Boolean
                : DataKind.String;

            var header = values[0].ToString();
            var name = string.IsNullOrWhiteSpace(header) ? $"col{i}" : header;
            var unique = name;
            for (var n = 0; !used.Add(unique); n++)
                unique = $"{name}_{n:00}";
            columns[i] = new TypedColumn(i, unique, kind);
        }

        if (label is null)
            return columns;

        var labelIndex = Array.FindIndex(columns, c => c.Name == label);
        if (labelIndex < 0)
            throw new ArgumentException($"Specified label column '{label}' was not found.");
        if (AllBoolean(rows.Select(r => r[labelIndex]).ToArray()))
            columns[labelIndex] = columns[labelIndex] with { Kind = DataKind.Boolean };

        return columns;

        static bool AllBoolean(ReadOnlyMemory<char>[] values) =>
            values.Skip(1).All(v => !v.IsEmpty && ColumnLayoutValues.IsBoolean(v.Span));
    }

    private static ColumnLayoutPurpose[] InferPurposes(MLContext mlContext, IDataView typed, TypedColumn[] columns, string? label)
    {
        var data = mlContext.Data.TakeRows(typed, PurposeRows);
        var purposes = new ColumnLayoutPurpose?[columns.Length];
        var toInfer = columns.Where(c => c.Name != label).ToArray();
        if (label is not null)
            purposes[Array.FindIndex(columns, c => c.Name == label)] = ColumnLayoutPurpose.Label;

        foreach (var column in toInfer.Where(c => c.Kind == DataKind.String))
        {
            var values = StringValues(data, column.Index);
            long length = 0;
            var spaces = 0;
            var seen = new HashSet<string>();
            foreach (var value in values)
            {
                length += value.Length;
                seen.Add(value);
                spaces += value.Count(ch => ch == ' ');
            }

            double distinct = 1.0 * seen.Count / values.Count;
            if (distinct < 0.7)
                purposes[column.Index] = ColumnLayoutPurpose.Categorical;
            else if (distinct >= 0.85 && (1.0 * length / values.Count > 30 || 1.0 * spaces / values.Count >= 1 || toInfer.Length == 1))
                purposes[column.Index] = ColumnLayoutPurpose.Text;
        }

        foreach (var column in toInfer)
            purposes[column.Index] ??= column.Kind == DataKind.String ? ColumnLayoutPurpose.Ignore : ColumnLayoutPurpose.Numeric;

        return [.. purposes.Select(p => p!.Value)];
    }

    private static List<string> StringValues(IDataView data, int index)
    {
        var values = new List<string>();
        var column = data.Schema[index];
        using var cursor = data.GetRowCursor([column]);
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(column);
        ReadOnlyMemory<char> value = default;
        while (cursor.MoveNext())
        {
            getter(ref value);
            values.Add(value.ToString());
        }
        return values;
    }

    // Columns of one kind and purpose fold together, except categorical, text and ignored ones, which
    // each stay their own. A group of one keeps its header name; a larger one is named for its purpose.
    private static List<Group> GroupColumns(TypedColumn[] columns, ColumnLayoutPurpose[] purposes)
    {
        var groups = new List<Group>();
        var keyed = columns
            .Select((column, i) => (column, purpose: purposes[i]))
            .GroupBy(t => (t.column.Kind, t.purpose, own: StaysItsOwn(t.purpose) ? t.column.Index : -1));
        foreach (var g in keyed)
        {
            var members = g.ToList();
            var name = members.Count == 1 ? members[0].column.Name : UnusedName(GroupName(g.Key.purpose, g.Key.Kind), groups);
            groups.Add(new Group(name, g.Key.Kind, g.Key.purpose, Ranges([.. members.Select(m => m.column.Index)])));
        }
        return groups;

        static bool StaysItsOwn(ColumnLayoutPurpose purpose) =>
            purpose is ColumnLayoutPurpose.Categorical or ColumnLayoutPurpose.Text or ColumnLayoutPurpose.Ignore;

        static string UnusedName(string prefix, List<Group> previous)
        {
            var name = prefix;
            for (var i = 1; previous.Any(p => p.Name == name); i++)
                name = $"{prefix}{i:00}";
            return name;
        }
    }

    private static string GroupName(ColumnLayoutPurpose purpose, DataKind kind) => purpose switch
    {
        ColumnLayoutPurpose.Numeric => kind == DataKind.Boolean ? "BooleanFeatures" : "Features",
        ColumnLayoutPurpose.Label => "Label",
        _ => purpose.ToString(),
    };

    private static TextLoader.Range[] Ranges(int[] indices)
    {
        Array.Sort(indices);
        var ranges = new List<TextLoader.Range>();
        var current = new TextLoader.Range(indices[0]);
        foreach (var index in indices.Skip(1))
        {
            if (index == current.Max + 1)
            {
                current.Max++;
            }
            else
            {
                ranges.Add(current);
                current = new TextLoader.Range(index);
            }
        }
        ranges.Add(current);
        return [.. ranges];
    }
}
