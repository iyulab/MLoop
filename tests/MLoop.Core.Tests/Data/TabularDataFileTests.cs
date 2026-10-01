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
}
