using System.Text;
using CsvHelper;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// What a delimited table means to MLoop whichever CSV library reads it. The library's own defaults moved with its major
/// versions (the separator is detected, a malformed quote is recovered, a record of another width is refused when written);
/// these pin the dialect <see cref="CsvDialect"/> promises.
/// </summary>
public class CsvDialectTests
{
    private static CsvReader Open(string text, CsvReaderOptions options)
        => CsvReader.Create(new StringReader(text), options);

    [Fact]
    public void The_separator_is_the_one_given_and_is_never_detected()
    {
        // Read as a comma table, a semicolon file is one column headed by its whole first line; the library would detect ';'.
        using var csv = Open("a;b\n1;2\n", CsvDialect.Reader());
        csv.ReadHeader();

        Assert.Equal(["a;b"], csv.Headers);
    }

    [Fact]
    public void A_separator_other_than_a_comma_is_read_when_it_is_asked_for()
    {
        using var csv = Open("a\tb\n1\t2\n", CsvDialect.Reader(separator: '\t'));
        csv.ReadHeader();
        Assert.Equal(["a", "b"], csv.Headers);
        Assert.True(csv.Read());
        Assert.Equal(["1", "2"], CsvDialect.Record(csv));
    }

    [Fact]
    public void A_short_row_has_fewer_cells_and_a_missing_cell_reads_empty()
    {
        using var csv = Open("a,b,c\n1,2\n", CsvDialect.Reader());
        csv.ReadHeader();
        Assert.True(csv.Read());

        Assert.Equal(["1", "2"], CsvDialect.Record(csv));
        Assert.Equal("", csv.GetString(2));
    }

    [Fact]
    public void Without_a_header_nothing_is_taken_for_one()
    {
        using var csv = Open("1,2\n3,4\n", CsvDialect.Reader(hasHeader: false));

        Assert.True(csv.Read());
        Assert.Equal(["1", "2"], CsvDialect.Record(csv));
        Assert.True(csv.Read());
        Assert.Equal(["3", "4"], CsvDialect.Record(csv));
        Assert.False(csv.Read());
    }

    [Fact]
    public void Trimming_removes_the_spaces_around_a_value_and_keeps_those_inside_quotes()
    {
        using var csv = Open("a,b\n  x  ,\"  y  \"\n", CsvDialect.Reader(trim: true));
        csv.ReadHeader();
        Assert.True(csv.Read());

        Assert.Equal(["x", "  y  "], CsvDialect.Record(csv));
    }

    [Theory]
    [InlineData("a,b\nab\"cd,2\n")]
    [InlineData("a,b\n\"abc,2\n3,4\n")]
    [InlineData("a,b\n\"x\"y,2\n")]
    public void A_malformed_quote_is_recovered_unless_the_caller_asks_for_it_to_be_refused(string text)
    {
        using (var csv = Open(text, CsvDialect.Reader()))
        {
            csv.ReadHeader();
            while (csv.Read()) { } // does not throw
        }

        using var strict = Open(text, CsvDialect.Reader(refuseMalformedQuotes: true));
        strict.ReadHeader();
        Assert.ThrowsAny<FormatException>(() => { while (strict.Read()) { } });
    }

    [Fact]
    public void A_blank_line_is_not_a_record()
    {
        using var csv = Open("a,b\n1,2\n\n3,4\n\n", CsvDialect.Reader());
        csv.ReadHeader();
        var rows = new List<string[]>();
        while (csv.Read())
            rows.Add(CsvDialect.Record(csv));

        Assert.Equal(2, rows.Count);
    }

    private static string Write(Action<CsvWriter> write)
    {
        using var buffer = new StringWriter { NewLine = "\n" };
        using (var csv = CsvWriter.Create(buffer, CsvDialect.Writer()))
            write(csv);
        return buffer.ToString();
    }

    [Fact]
    public void A_value_is_quoted_when_it_needs_it_or_when_it_starts_or_ends_with_a_space()
    {
        var text = Write(csv =>
        {
            foreach (var field in new[] { "plain", " lead", "trail ", "c,d", "q\"uote", "l\nf" })
                csv.WriteField(field);
            csv.EndRecord();
        });

        Assert.Equal("plain,\" lead\",\"trail \",\"c,d\",\"q\"\"uote\",\"l\nf\"\r\n", text);
    }

    [Fact]
    public void A_value_that_looks_like_a_formula_is_written_as_it_is()
    {
        var text = Write(csv =>
        {
            foreach (var field in new[] { "=1+1", "-5", "@x", "+3" })
                csv.WriteField(field);
            csv.EndRecord();
        });

        Assert.Equal("=1+1,-5,@x,+3\r\n", text);
    }

    [Fact]
    public void A_record_of_another_width_is_written_as_it_was_given()
    {
        // The rows written are the rows given — a slice of a file keeps a ragged row as it was.
        var text = Write(csv =>
        {
            csv.WriteField("a"); csv.WriteField("b"); csv.WriteField("c"); csv.EndRecord();
            csv.WriteField("1"); csv.WriteField("2"); csv.EndRecord();
            csv.WriteField("3"); csv.WriteField("4"); csv.WriteField("5"); csv.WriteField("6"); csv.EndRecord();
        });

        Assert.Equal("a,b,c\r\n1,2\r\n3,4,5,6\r\n", text);
    }

    [Fact]
    public void A_one_column_row_with_an_empty_value_is_written_so_that_it_is_still_a_row()
    {
        var text = Write(csv =>
        {
            csv.WriteField("a"); csv.EndRecord();
            csv.WriteField(""); csv.EndRecord();
            csv.WriteField("3"); csv.EndRecord();
        });

        using var csv = Open(text, CsvDialect.Reader());
        csv.ReadHeader();
        var rows = new List<string[]>();
        while (csv.Read())
            rows.Add(CsvDialect.Record(csv));

        Assert.Equal(2, rows.Count);
        Assert.Equal([""], rows[0]);
    }

    [Fact]
    public void A_text_round_trips_through_write_and_read()
    {
        var cells = new[] { "한글", " 앞뒤 ", "a,b", "큰\"따옴표", "줄\n바꿈", "", "=sum(A1)" };
        var text = Write(csv =>
        {
            csv.WriteField("c0"); csv.WriteField("c1"); csv.WriteField("c2"); csv.WriteField("c3"); csv.WriteField("c4"); csv.WriteField("c5"); csv.WriteField("c6"); csv.EndRecord();
            foreach (var cell in cells)
                csv.WriteField(cell);
            csv.EndRecord();
        });

        using var reader = Open(text, CsvDialect.Reader());
        reader.ReadHeader();
        Assert.True(reader.Read());
        Assert.Equal(cells, CsvDialect.Record(reader));
    }

    [Fact]
    public void A_file_with_a_byte_order_mark_has_a_header_without_it()
    {
        var bytes = new UTF8Encoding(true).GetPreamble().Concat(Encoding.UTF8.GetBytes("a,b\n1,2\n")).ToArray();
        using var csv = CsvReader.Create(new StreamReader(new MemoryStream(bytes), Encoding.UTF8, detectEncodingFromByteOrderMarks: true), CsvDialect.Reader());
        csv.ReadHeader();

        Assert.Equal(["a", "b"], csv.Headers);
    }
}
