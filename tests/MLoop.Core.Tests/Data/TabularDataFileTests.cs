using MLoop.Core.Data;
using Parquet.Serialization;

namespace MLoop.Core.Tests.Data;

public class TabularDataFileTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop-tabular-{Guid.NewGuid():N}");

    public TabularDataFileTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, true);
    }

    public sealed class Scores
    {
        public double Label { get; set; }
    }

    public sealed class Row
    {
        public string Text { get; set; } = "";
        public Scores Labels { get; set; } = new();
    }

    private async Task<string> WriteParquetAsync()
    {
        var path = Path.Combine(_dir, "rows.parquet");
        await ParquetSerializer.SerializeAsync(new[]
        {
            new Row { Text = "좋아요", Labels = new() { Label = 4.5 } },
            new Row { Text = "별로, 그냥", Labels = new() { Label = 1.25 } }
        }, path);
        return path;
    }

    [Fact]
    public async Task A_csv_file_is_returned_as_it_is()
    {
        var csv = Path.Combine(_dir, "data.csv");
        await File.WriteAllTextAsync(csv, "a,b\n1,2\n");

        Assert.Equal(csv, await TabularDataFile.AsCsvAsync(csv));
    }

    [Fact]
    public async Task A_parquet_file_becomes_a_csv_with_its_struct_leaves_as_dotted_columns()
    {
        var converted = await TabularDataFile.AsCsvAsync(await WriteParquetAsync());

        // Readers downstream route on the extension, so the copy must say what it now is.
        Assert.Equal(".csv", Path.GetExtension(converted));
        var lines = (await File.ReadAllLinesAsync(converted)).Select(l => l.TrimStart('﻿')).ToArray();
        Assert.Equal("Text,Labels.Label", lines[0]);
        Assert.Equal("좋아요,4.5", lines[1]);
        Assert.Equal("\"별로, 그냥\",1.25", lines[2]);
    }

    [Fact]
    public async Task One_file_is_converted_once_however_many_steps_ask()
    {
        var parquet = await WriteParquetAsync();

        var first = await TabularDataFile.AsCsvAsync(parquet);
        var second = await TabularDataFile.AsCsvAsync(parquet);

        Assert.Equal(first, second);
    }

    [Theory]
    [InlineData("data.csv", false)]
    [InlineData("data.txt", false)]
    [InlineData("data.parquet", true)]
    [InlineData("data.xlsx", true)]
    [InlineData("data.json", true)]
    [InlineData("data.tsv", true)]
    public void Only_formats_other_than_delimited_text_are_converted(string path, bool expected) =>
        Assert.Equal(expected, TabularDataFile.NeedsConversion(path));

    [Fact]
    public async Task A_record_path_reads_the_rows_of_a_nested_json_array()
    {
        var path = Path.Combine(_dir, "squad.json");
        File.WriteAllText(path, """
            {"version":"1","data":[{"paragraphs":[{"context":"서울은 수도다","qas":[
              {"question":"수도는?","answers":[{"text":"서울","answer_start":0}]}]}]}]}
            """);

        var csv = await TabularDataFile.AsCsvAsync(path, "data.paragraphs.qas.answers");
        var lines = File.ReadAllLines(csv);

        Assert.Equal(2, lines.Length);
        Assert.Contains("paragraphs.context", lines[0]);
        Assert.Contains("qas.question", lines[0]);
        Assert.Contains("서울은 수도다", lines[1]);
    }

    [Fact]
    public async Task A_record_path_for_a_file_that_is_not_json_is_refused_not_ignored()
    {
        var path = Path.Combine(_dir, "rows.csv");
        File.WriteAllLines(path, ["a,b", "1,2"]);

        var ex = await Assert.ThrowsAsync<ArgumentException>(() => TabularDataFile.AsCsvAsync(path, "data"));

        Assert.Contains("--records", ex.Message);
    }

    [Fact]
    public async Task A_file_holding_one_record_is_a_one_row_table()
    {
        var path = Path.Combine(_dir, "reading.json");
        File.WriteAllText(path, """{"id": "00002", "volts": 3.295, "temp.1": 26.9}""");

        var lines = File.ReadAllLines(await TabularDataFile.AsCsvAsync(path));

        Assert.Equal(2, lines.Length);
        Assert.Contains("temp.1", lines[0]);
    }

    [Fact]
    public async Task Records_is_suggested_only_when_there_is_a_path_to_give()
    {
        var nested = Path.Combine(_dir, "nested.json");
        File.WriteAllText(nested, """{"data": [{"x": 1}]}""");
        var scalars = Path.Combine(_dir, "scalars.json");
        File.WriteAllText(scalars, "[1, 2, 3]");

        var withPath = await Assert.ThrowsAnyAsync<Exception>(() => TabularDataFile.AsCsvAsync(nested));
        var withoutPath = await Assert.ThrowsAnyAsync<Exception>(() => TabularDataFile.AsCsvAsync(scalars));

        Assert.Contains("--records", withPath.Message);
        Assert.Contains("'data'", withPath.Message);
        Assert.DoesNotContain("--records", withoutPath.Message);
    }

    [Fact]
    public async Task A_folder_of_data_files_is_one_table_in_file_name_order()
    {
        var folder = Path.Combine(_dir, "parts");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "part_002.json"), """{"x": 2}""");
        File.WriteAllText(Path.Combine(folder, "part_001.json"), """{"x": 1}""");
        File.WriteAllText(Path.Combine(folder, ".done"), "");

        Assert.True(TabularDataFile.Exists(folder));
        Assert.True(TabularDataFile.NeedsConversion(folder));
        var csv = await TabularDataFile.AsCsvAsync(folder);

        Assert.Equal(".csv", Path.GetExtension(csv));
        Assert.Equal(["x", "1", "2"], File.ReadAllLines(csv).Select(l => l.Trim('\uFEFF')));
        Assert.Equal(16, TabularDataFile.SizeOf(folder)); // the two data files, not the marker
    }

    [Fact]
    public void A_folder_without_data_files_is_not_data()
    {
        var folder = Path.Combine(_dir, "empty");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "notes.md"), "# about");

        Assert.False(TabularDataFile.Exists(folder));
        Assert.False(TabularDataFile.NeedsConversion(folder));
    }

    [Fact]
    public async Task Records_apply_to_a_folder_of_json_files_and_are_refused_for_one_of_csv()
    {
        var folder = Path.Combine(_dir, "csvs");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "a.csv"), "x\n1\n");

        await Assert.ThrowsAsync<ArgumentException>(() => TabularDataFile.AsCsvAsync(folder, "data"));
    }

    // ── delimited text whose Hangul is decomposed (NFD) is read as the same text, composed ──

    private static string Nfd(string text) => text.Normalize(System.Text.NormalizationForm.FormD);

    private string WriteText(string name, string content, System.Text.Encoding encoding)
    {
        var path = Path.Combine(_dir, name);
        File.WriteAllText(path, content, encoding);
        return path;
    }

    [Fact]
    public async Task A_file_already_composed_is_returned_as_it_is()
    {
        var plain = WriteText("composed.csv", "지역,면적\n서울,1\n", new System.Text.UTF8Encoding(false));
        var ascii = WriteText("ascii.csv", "a,b\n1,2\n", new System.Text.UTF8Encoding(false));
        var empty = WriteText("empty.csv", "", new System.Text.UTF8Encoding(false));

        Assert.Equal(plain, await TabularDataFile.AsCsvAsync(plain));
        Assert.Equal(ascii, await TabularDataFile.AsCsvAsync(ascii));
        Assert.Equal(empty, await TabularDataFile.AsCsvAsync(empty));
    }

    [Fact]
    public async Task A_decomposed_file_becomes_a_copy_with_the_text_composed_and_nothing_else_changed()
    {
        // CRLF endings, a byte-order mark and a quoted field with a comma and a doubled quote: all of it stays.
        var original = $"{Nfd("지역")},{Nfd("면적")}\r\n\"{Nfd("서울, 강남")}\",1\r\n{Nfd("부산")},\"a \"\"b\"\"\"\r\n";
        var path = WriteText("decomposed.csv", original, new System.Text.UTF8Encoding(true));

        var result = await TabularDataFile.AsCsvAsync(path);

        Assert.NotEqual(path, result);
        Assert.EndsWith(".csv", result);
        var bytes = await File.ReadAllBytesAsync(result);
        Assert.Equal([0xEF, 0xBB, 0xBF], bytes[..3]);                       // the byte-order mark stays
        var text = System.Text.Encoding.UTF8.GetString(bytes[3..]);
        Assert.Equal(original.Normalize(System.Text.NormalizationForm.FormC), text);
        Assert.Contains("\r\n", text);                                      // line endings stay
        Assert.Equal(original, await File.ReadAllTextAsync(path));          // the file named is not touched
        Assert.Equal(result, await TabularDataFile.AsCsvAsync(path));       // decided once per process
    }

    [Fact]
    public async Task A_file_that_is_not_utf8_is_never_rewritten()
    {
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        var cp949 = System.Text.Encoding.GetEncoding(949);
        var path = WriteText("cp949.csv", "지역,면적\n서울,1\n", cp949);

        Assert.Equal(path, await TabularDataFile.AsCsvAsync(path));
    }

    [Fact]
    public async Task A_decomposed_line_far_into_a_large_file_is_found()
    {
        var lines = Enumerable.Range(0, 20000).Select(i => $"서울,{i}").ToList();
        lines.Insert(15000, $"{Nfd("부산")},0");
        var path = WriteText("large.csv", "지역,값\n" + string.Join('\n', lines) + "\n", new System.Text.UTF8Encoding(false));

        var result = await TabularDataFile.AsCsvAsync(path);

        Assert.NotEqual(path, result);
        // Ordinal: the culture-aware comparison xunit's Contains defaults to treats the two forms as equal.
        Assert.False((await File.ReadAllTextAsync(result)).Contains(Nfd("부산"), StringComparison.Ordinal));
        Assert.Equal(20002, (await File.ReadAllLinesAsync(result)).Length);
    }
}
