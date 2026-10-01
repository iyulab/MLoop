using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// Rows whose label is empty are dropped by default wherever the label is a target. It was the default
/// for classification only, and a recommendation set's unrated visits (22% of rows) trained and scored a
/// model at R² 0.54 that scored −0.64 once they were left out.
/// </summary>
public class MissingLabelDefaultTests
{
    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("regression")]
    [InlineData("recommendation")]
    [InlineData("ranking")]
    [InlineData("ner")]
    [InlineData("question-answering")]
    public void A_task_whose_label_is_a_target_drops_rows_without_one(string task) =>
        Assert.True(TrainCommand.DropsMissingLabelsByDefault(task));

    [Theory]
    [InlineData("forecasting")]
    [InlineData("time-series-anomaly")]
    [InlineData("clustering")]
    [InlineData("anomaly-detection")]
    [InlineData("image-classification")]
    [InlineData("object-detection")]
    public void A_series_an_unsupervised_task_or_an_image_directory_keeps_its_rows(string task) =>
        Assert.False(TrainCommand.DropsMissingLabelsByDefault(task));
}
