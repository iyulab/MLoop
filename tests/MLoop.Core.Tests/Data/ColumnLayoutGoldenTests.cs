using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// The column layout a saved model's input contract rests on, pinned byte for byte: what ML.NET AutoML's
/// <c>InferColumns</c> — called as MLoop calls it at every site (a file, the label's name, <c>','</c>) —
/// infers for a corpus of tables, rendered field by field into <c>Golden/column-layout.txt</c>. A model
/// reads its input through the <c>TextLoader.Options</c> this produces, so the inference that replaces
/// AutoML's must reproduce this file exactly. Each table exercises one rule: adjacent numbers folded into
/// one vector, the boolean label, quoting, rows over several lines, empty and special numbers, a type that
/// changes only past the sampled rows, the purpose a text column is given.
/// </summary>
/// <remarks>
/// <c>MLOOP_UPDATE_GOLDEN=1</c> rewrites the file from the current inference. Do that only when the
/// layout is meant to change — and then every saved model's input contract changes with it.
/// </remarks>
[Collection("FileSystem")]
public sealed class ColumnLayoutGoldenTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_layout_{Guid.NewGuid():N}");

    public ColumnLayoutGoldenTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private sealed record Table(string Name, string Label, IReadOnlyList<string> Lines);

    private static IEnumerable<Table> Corpus()
    {
        static string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);
        var rng = new Random(41);

        yield return new Table("adjacent-numbers-bool-label", "label",
            ["a,b,c,d,label", .. Enumerable.Range(0, 60).Select(i => $"{F(rng.NextDouble())},{i},{F(rng.NextDouble() * 100)},{i % 7},{(i % 3 == 0 ? "true" : "false")}")]);

        yield return new Table("numbers-split-by-a-category", "label",
            ["x,y,colour,z,w,label", .. Enumerable.Range(0, 60).Select(i => $"{i},{F(rng.NextDouble())},{(i % 3) switch { 0 => "red", 1 => "green", _ => "blue" }},{i * 2},{F(rng.NextDouble())},{(i % 2 == 0 ? "yes" : "no")}")]);

        string[] words = ["the", "machine", "stopped", "after", "a", "long", "run", "and", "then", "restarted", "slowly", "today"];
        yield return new Table("free-text-and-category", "label",
            ["note,kind,score,label", .. Enumerable.Range(0, 80).Select(i => $"{string.Join(' ', Enumerable.Range(0, 6).Select(_ => words[rng.Next(words.Length)]))} {i},{(i % 4 == 0 ? "A" : "B")},{F(rng.NextDouble())},{(i % 2 == 0 ? "1" : "0")}")]);

        yield return new Table("quoted-commas", "label",
            ["name,amount,label", .. Enumerable.Range(0, 40).Select(i => $"\"Smith, J{i}\",{i},{(i % 2 == 0 ? "true" : "false")}")]);

        yield return new Table("quoted-multiline", "label",
            ["comment,amount,label", .. Enumerable.Range(0, 40).Select(i => i % 5 == 0 ? $"\"first line {i}\nsecond line\",{i},{(i % 2 == 0 ? "true" : "false")}" : $"plain {i},{i},{(i % 2 == 0 ? "true" : "false")}")]);

        yield return new Table("empty-and-special-numbers", "label",
            ["a,b,c,label", .. Enumerable.Range(0, 50).Select(i => $"{(i % 6 == 0 ? "" : F(i * 0.5))},{(i % 10) switch { 0 => "NaN", 1 => "Infinity", 2 => "+3", 3 => "-Infinity", _ => F(i) }},{i},{(i % 2 == 0 ? "true" : "false")}")]);

        yield return new Table("boolean-feature-with-a-gap", "label",
            ["flag,other_flag,v,label", .. Enumerable.Range(0, 50).Select(i => $"{(i % 2 == 0 ? "true" : "false")},{(i % 9 == 0 ? "" : i % 2 == 0 ? "True" : "False")},{i},{(i % 3 == 0 ? "true" : "false")}")]);

        yield return new Table("zero-one-label", "outcome",
            ["a,b,outcome", .. Enumerable.Range(0, 50).Select(i => $"{i},{F(rng.NextDouble())},{i % 2}")]);

        yield return new Table("numeric-label", "price",
            ["size,rooms,price", .. Enumerable.Range(0, 50).Select(i => $"{i * 10},{i % 5},{F(1000 + i * 37.5)}")]);

        yield return new Table("string-label-multiclass", "tier",
            ["a,b,tier", .. Enumerable.Range(0, 60).Select(i => $"{i},{F(rng.NextDouble())},{(i % 3) switch { 0 => "gold", 1 => "silver", _ => "bronze" }}")]);

        // Numbers for 10,000 rows, then words: whether the type is read from every row or a sample.
        yield return new Table("type-changes-past-the-sample", "label",
            ["a,late,label", .. Enumerable.Range(0, 10_050).Select(i => $"{i},{(i < 10_000 ? i.ToString(CultureInfo.InvariantCulture) : "word")},{(i % 2 == 0 ? "true" : "false")}")]);

        // A column of distinct strings for 1,000 rows, then a few repeated values.
        yield return new Table("purpose-past-a-thousand-rows", "label",
            ["code,v,label", .. Enumerable.Range(0, 1_200).Select(i => $"{(i < 1_000 ? $"id{i}" : "shared")},{i},{(i % 2 == 0 ? "true" : "false")}")]);

        yield return new Table("leading-spaces", "label",
            ["a,b,label", .. Enumerable.Range(0, 30).Select(i => $" {i}, x{i % 3},{(i % 2 == 0 ? "true" : "false")}")]);

        yield return new Table("wide-numbers", "label",
            [string.Join(',', Enumerable.Range(0, 40).Select(c => $"f{c}")) + ",label",
             .. Enumerable.Range(0, 40).Select(i => string.Join(',', Enumerable.Range(0, 40).Select(c => (i * c % 17).ToString(CultureInfo.InvariantCulture))) + $",{(i % 2 == 0 ? "true" : "false")}")]);

        yield return new Table("header-only", "label", ["a,b,label"]);
    }

    [Fact]
    public void InferredLayouts_MatchTheGoldenFile()
    {
        var rendered = Render();

        if (Environment.GetEnvironmentVariable("MLOOP_UPDATE_GOLDEN") == "1")
        {
            File.WriteAllText(SourceGoldenPath(), rendered);
            return;
        }

        // Read from the build output, where the project copies it: a deterministic build (CI) maps the
        // source path a test could name to /_/, which does not exist.
        var golden = Path.Combine(AppContext.BaseDirectory, "Golden", "column-layout.txt");
        Assert.True(File.Exists(golden), $"No golden file at {golden} — run once with MLOOP_UPDATE_GOLDEN=1.");
        Assert.Equal(File.ReadAllText(golden).ReplaceLineEndings("\n"), rendered);
    }

    private string Render()
    {
        var text = new StringBuilder();
        foreach (var table in Corpus())
        {
            var path = Path.Combine(_dir, table.Name + ".csv");
            File.WriteAllText(path, string.Join("\n", table.Lines) + "\n");
            text.Append("## ").Append(table.Name).Append(" (label: ").Append(table.Label).Append(")\n");
            try
            {
                var inference = new MLContext(seed: 1).Auto().InferColumns(path, labelColumnName: table.Label, separatorChar: ',');
                Describe(text, inference);
            }
            catch (Exception ex)
            {
                text.Append("throws ").Append(ex.GetType().Name).Append(": ").Append(ex.Message.ReplaceLineEndings(" ")).Append('\n');
            }
            text.Append('\n');
        }
        return text.ToString();
    }

    private static void Describe(StringBuilder text, ColumnInferenceResults inference)
    {
        var o = inference.TextLoaderOptions;
        text.Append(CultureInfo.InvariantCulture,
            $"options: separators=[{string.Join("", o.Separators)}] header={o.HasHeader} quoting={o.AllowQuoting} sparse={o.AllowSparse} multiline={o.ReadMultilines} trim={o.TrimWhitespace} nan-for-missing={o.MissingRealsAsNaNs} escape-quotes={o.EscapeChar}\n");
        foreach (var column in o.Columns)
        {
            var sources = string.Join(" ", column.Source.Select(r =>
                r.Min == r.Max ? r.Min.ToString(CultureInfo.InvariantCulture) : $"{r.Min}-{(r.Max is { } max ? max.ToString(CultureInfo.InvariantCulture) : "*")}"));
            text.Append(CultureInfo.InvariantCulture, $"column {column.Name}: {column.DataKind} [{sources}]{(column.KeyCount is null ? "" : " key")}\n");
        }

        var info = inference.ColumnInformation;
        void Roles(string role, IEnumerable<string> names)
        {
            var list = names.ToList();
            if (list.Count > 0)
                text.Append(role).Append(": ").Append(string.Join(", ", list)).Append('\n');
        }
        text.Append("label: ").Append(info.LabelColumnName).Append('\n');
        Roles("numeric", info.NumericColumnNames);
        Roles("categorical", info.CategoricalColumnNames);
        Roles("text", info.TextColumnNames);
        Roles("ignored", info.IgnoredColumnNames);
        Roles("image", info.ImagePathColumnNames);
        if (info.ExampleWeightColumnName is { } weight)
            text.Append("weight: ").Append(weight).Append('\n');
        if (info.SamplingKeyColumnName is { } key)
            text.Append("sampling key: ").Append(key).Append('\n');
    }

    // Where an update writes: the source tree, on a machine that builds from it.
    private static string SourceGoldenPath([CallerFilePath] string source = "")
        => Path.Combine(Path.GetDirectoryName(source)!, "Golden", "column-layout.txt");
}
