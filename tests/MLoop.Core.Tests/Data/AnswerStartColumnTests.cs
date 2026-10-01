using Microsoft.ML;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class AnswerStartColumnTests
{
    private sealed class Row
    {
        public string Passage { get; set; } = "";
        public string Question { get; set; } = "";
        public string Answer { get; set; } = "";
        public float Id { get; set; }
        public float Offset { get; set; }
    }

    private static IDataView Rows(params Row[] rows) => new MLContext(seed: 0).Data.LoadFromEnumerable(rows);

    private static string? Find(IDataView data) =>
        AnswerStartColumn.Find(data, "Passage", "Answer", ["Passage", "Question", "Answer"]);

    [Fact]
    public void The_column_that_locates_the_answer_is_found_whatever_its_name()
    {
        var data = Rows(
            new Row { Passage = "서울은 한국의 수도이다", Answer = "수도", Id = 1, Offset = 8 },
            new Row { Passage = "부산에서 만났다", Answer = "부산", Id = 2, Offset = 0 });

        // `Id` is numeric too, and comes first; only `Offset` puts the passage's answer at its position.
        Assert.Equal("Offset", Find(data));
    }

    [Fact]
    public void No_column_is_found_when_no_position_gives_the_answer()
    {
        var data = Rows(new Row { Passage = "서울은 한국의 수도이다", Answer = "수도", Id = 1, Offset = 3 });

        Assert.Null(Find(data));
    }

    [Fact]
    public void A_position_pointing_at_the_space_a_trimmed_answer_lost_still_locates_it()
    {
        // Recorded as " 수도" at 7; the file is read with values trimmed, so the answer arrives as "수도".
        Assert.Equal(8, AnswerStartColumn.Locate("서울은 한국의 수도이다", "수도", 7));
        Assert.Equal(8, AnswerStartColumn.Locate("서울은 한국의 수도이다", "수도", 8));
        Assert.Null(AnswerStartColumn.Locate("서울은 한국의 수도이다", "수도", 5));
        Assert.Null(AnswerStartColumn.Locate("서울은 한국의 수도이다", "수도", 7.5));
    }

    [Fact]
    public void A_position_past_the_passage_does_not_count()
    {
        var data = Rows(new Row { Passage = "짧다", Answer = "짧다", Id = 0, Offset = 50 });

        Assert.Equal("Id", Find(data));
    }
}

public class AnswerStartColumnLoaderTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-qa-start-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void The_start_column_survives_the_loader_when_other_numeric_columns_sit_beside_it()
    {
        // A reading-comprehension file read by record path carries the enclosing items' fields — here a
        // numeric source and title — next to the answer's start. The loader must still give the start
        // its own column, or no column is left to locate the answer with.
        var path = Path.Combine(_dir, "qa.csv");
        var lines = new List<string> { "source,title,context,question,text,answer_start" };
        string[] passages = ["서울은 한국의 수도이다", "부산에서 친구를 만났다", "제주도는 섬이다 바다가 넓다"];
        string[] answers = ["수도", "친구", "바다"];
        for (var i = 0; i < 30; i++)
        {
            var p = passages[i % 3];
            var a = answers[i % 3];
            lines.Add($"{i % 4},{100 + i * 7},{p},질문 {i}는 무엇인가,{a},{p.IndexOf(a, StringComparison.Ordinal)}");
        }
        File.WriteAllLines(path, lines);

        var data = new CsvDataLoader(new MLContext(seed: 0)).LoadData(path, "text", "question-answering");

        Assert.Equal("answer_start",
            AnswerStartColumn.Find(data, "context", "text", ["context", "question", "text"]));
    }
}
