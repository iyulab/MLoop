using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;

namespace MLoop.Tests.ML;

/// <summary>
/// A NER label's classes are its tags. Counted as whole cells, N was the number of distinct
/// sentences, the quality gate's 1/N floor effectively zero, and a model that answers O for every
/// word was promoted (measured: micro 0.85 = the share of O, tag average 0.12).
/// </summary>
public class NerLabelSchemaTests : IDisposable
{
    private readonly string _file = Path.Combine(Path.GetTempPath(), $"mloop-ner-{Guid.NewGuid():N}.csv");

    public void Dispose()
    {
        if (File.Exists(_file)) File.Delete(_file);
    }

    [Fact]
    public void The_label_records_its_tags_their_count_and_the_share_of_the_commonest()
    {
        File.WriteAllText(_file,
            "Sentence,Label\n" +
            "a b c d,O O B-DATE O\n" +
            "e f,O B-TIME\n");
        var columns = new List<ColumnSchema>
        {
            new() { Name = "Sentence", DataType = "Text", Purpose = "Feature" },
            new() { Name = "Label", DataType = "Text", Purpose = "Label" }
        };

        TrainingEngine.CountLabelTags(_file, columns, ["Sentence", "Label"], "Label");

        var label = columns[1];
        Assert.Equal(["B-DATE", "B-TIME", "O"], label.CategoricalValues);
        Assert.Equal(3, label.UniqueValueCount);
        Assert.Equal(4.0 / 6, label.MajorityClassRatio!.Value, 6);
    }
}
