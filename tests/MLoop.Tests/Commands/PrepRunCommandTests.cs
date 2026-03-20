using MLoop.CLI.Commands;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands;

public class PrepRunCommandTests
{
    #region GetStepDetails

    [Fact]
    public void GetStepDetails_EmptyStep_ReturnsDash()
    {
        var step = new PrepStep { Type = "fill-missing" };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("-", result);
    }

    [Fact]
    public void GetStepDetails_WithColumns_ShowsColumns()
    {
        var step = new PrepStep
        {
            Type = "fill-missing",
            Columns = ["pH", "Temp", "Pressure"]
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("columns:", result);
        Assert.Contains("pH", result);
        Assert.Contains("Temp", result);
        Assert.Contains("Pressure", result);
    }

    [Fact]
    public void GetStepDetails_WithColumn_ShowsColumn()
    {
        var step = new PrepStep
        {
            Type = "extract-date",
            Column = "timestamp"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("column: timestamp", result);
    }

    [Fact]
    public void GetStepDetails_WithMethod_ShowsMethod()
    {
        var step = new PrepStep
        {
            Type = "fill-missing",
            Method = "mean"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("method: mean", result);
    }

    [Fact]
    public void GetStepDetails_WithMapping_ShowsEntryCount()
    {
        var step = new PrepStep
        {
            Type = "rename-columns",
            Mapping = new Dictionary<string, string>
            {
                ["old1"] = "new1",
                ["old2"] = "new2"
            }
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("mapping: 2 entries", result);
    }

    [Fact]
    public void GetStepDetails_WithOperator_ShowsOperatorAndValue()
    {
        var step = new PrepStep
        {
            Type = "filter-rows",
            Operator = "gt",
            Value = "100"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("op: gt 100", result);
    }

    [Fact]
    public void GetStepDetails_WithFormat_ShowsFormat()
    {
        var step = new PrepStep
        {
            Type = "parse-datetime",
            Format = "yyyy-MM-dd"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("format: yyyy-MM-dd", result);
    }

    [Fact]
    public void GetStepDetails_WithWindowSize_ShowsWindow()
    {
        var step = new PrepStep
        {
            Type = "rolling",
            WindowSize = 5
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("window: 5", result);
    }

    [Fact]
    public void GetStepDetails_WithWindowString_ShowsWindow()
    {
        var step = new PrepStep
        {
            Type = "resample",
            Window = "1H"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("window: 1H", result);
    }

    [Fact]
    public void GetStepDetails_WithTimeColumn_ShowsTime()
    {
        var step = new PrepStep
        {
            Type = "resample",
            TimeColumn = "timestamp"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("time: timestamp", result);
    }

    [Fact]
    public void GetStepDetails_WithRemoveOriginal_ShowsFlag()
    {
        var step = new PrepStep
        {
            Type = "extract-date",
            Column = "date",
            RemoveOriginal = true
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains("remove_original", result);
    }

    [Fact]
    public void GetStepDetails_MultipleParts_JoinsWithComma()
    {
        var step = new PrepStep
        {
            Type = "fill-missing",
            Columns = ["x1"],
            Method = "mean"
        };

        var result = PrepRunCommand.GetStepDetails(step);

        Assert.Contains(", ", result);
        Assert.Contains("columns:", result);
        Assert.Contains("method: mean", result);
    }

    #endregion
}
