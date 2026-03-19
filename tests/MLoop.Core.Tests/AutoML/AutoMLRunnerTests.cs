using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

public class AutoMLRunnerTests
{
    private readonly MLContext _mlContext = new MLContext(seed: 42);

    #region BuildColumnInformation Tests

    [Fact]
    public void BuildColumnInformation_TextOnlyDataset_ReturnsWithTextColumns()
    {
        // Arrange: text-only dataset (Content + Label)
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "This is a positive review", Label = true },
            new TextBinaryData { Content = "This is a negative review", Label = false },
            new TextBinaryData { Content = "Another positive text", Label = true },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Label", result.LabelColumnName);
        Assert.Contains("Content", result.TextColumnNames);
        Assert.Empty(result.NumericColumnNames);
        Assert.Empty(result.IgnoredColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_NumericOnlyDataset_ReturnsNull()
    {
        // Arrange: numeric-only dataset
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
            new NumericData { Feature1 = 3.0f, Feature2 = 4.0f, Label = false },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert: null means use existing behavior (no text columns)
        Assert.Null(result);
    }

    [Fact]
    public void BuildColumnInformation_MixedTextAndNumeric_ReturnsBothClassified()
    {
        // Arrange: mixed dataset
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MixedData { Description = "Good product", Price = 29.99f, Label = true },
            new MixedData { Description = "Bad product", Price = 9.99f, Label = false },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Description", result.TextColumnNames);
        Assert.Contains("Price", result.NumericColumnNames);
        Assert.Equal("Label", result.LabelColumnName);
    }

    [Fact]
    public void BuildColumnInformation_LabelColumnExcluded_NotInAnyFeatureList()
    {
        // Arrange
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "text", Label = true },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.DoesNotContain("Label", result.TextColumnNames);
        Assert.DoesNotContain("Label", result.NumericColumnNames);
        Assert.DoesNotContain("Label", result.IgnoredColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_MultipleTextColumns_AllClassified()
    {
        // Arrange: multiple text columns
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MultiTextData { Title = "Great", Body = "Detailed review text", Label = true },
            new MultiTextData { Title = "Bad", Body = "Short negative review", Label = false },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Title", result.TextColumnNames);
        Assert.Contains("Body", result.TextColumnNames);
        Assert.Equal(2, result.TextColumnNames.Count);
    }

    [Fact]
    public void BuildColumnInformation_CaseInsensitiveLabelMatch()
    {
        // Arrange
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "text", Label = true },
        });

        // Act — label column name case doesn't match property name exactly
        var result = AutoMLRunner.BuildColumnInformation(data, "label");

        // Assert: Label should still be excluded from features
        Assert.NotNull(result);
        Assert.Single(result.TextColumnNames);
        Assert.Contains("Content", result.TextColumnNames);
    }

    #endregion

    #region ColumnOverride Tests

    [Fact]
    public void BuildColumnInformation_WithTextOverride_ForcesTextClassification()
    {
        // Arrange: numeric-only dataset, but override Feature1 as text
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
        });
        // Note: ML.NET loads float as NumberDataViewType, but override should still add to correct list
        var overrides = new Dictionary<string, string> { ["Feature1"] = "text" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert: non-null because override was applied
        Assert.NotNull(result);
        Assert.Contains("Feature1", result.TextColumnNames);
        Assert.Contains("Feature2", result.NumericColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_WithIgnoreOverride_ExcludesColumn()
    {
        // Arrange: text dataset, but override Content as ignore
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "some text", Label = true },
        });
        var overrides = new Dictionary<string, string> { ["Content"] = "ignore" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert
        Assert.NotNull(result);
        Assert.DoesNotContain("Content", result.TextColumnNames);
        Assert.Contains("Content", result.IgnoredColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_WithCategoricalOverride_ClassifiesAsCategorical()
    {
        // Arrange: text dataset, but override Content as categorical
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "category_A", Label = true },
        });
        var overrides = new Dictionary<string, string> { ["Content"] = "categorical" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert
        Assert.NotNull(result);
        Assert.DoesNotContain("Content", result.TextColumnNames);
        Assert.Contains("Content", result.CategoricalColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_WithNumericOverride_ClassifiesAsNumeric()
    {
        // Arrange: text dataset, but override Description as numeric
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MixedData { Description = "text", Price = 10.0f, Label = true },
        });
        var overrides = new Dictionary<string, string> { ["Description"] = "numeric" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert: Description forced to numeric — no text columns remain,
        // so ColumnInformation is null (AutoML default behavior handles numeric fine)
        Assert.Null(result);
    }

    [Fact]
    public void BuildColumnInformation_WithMixedOverrides_AllClassifiedCorrectly()
    {
        // Arrange: dataset with text + numeric, apply mixed overrides
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MultiTextData { Title = "title", Body = "body text", Label = true },
        });
        var overrides = new Dictionary<string, string>
        {
            ["Title"] = "categorical",  // force text → categorical
            // Body stays as text (no override)
        };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Title", result.CategoricalColumnNames);
        Assert.Contains("Body", result.TextColumnNames);
        Assert.DoesNotContain("Title", result.TextColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_NoOverrides_BehavesAsOriginal()
    {
        // Arrange
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
        });

        // Act: pass empty overrides
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: new Dictionary<string, string>());

        // Assert: same as no overrides — returns null for numeric-only
        Assert.Null(result);
    }

    #endregion

    #region Test Data Classes

    private class TextBinaryData
    {
        public string Content { get; set; } = "";
        public bool Label { get; set; }
    }

    private class NumericData
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public bool Label { get; set; }
    }

    private class MixedData
    {
        public string Description { get; set; } = "";
        public float Price { get; set; }
        public bool Label { get; set; }
    }

    private class MultiTextData
    {
        public string Title { get; set; } = "";
        public string Body { get; set; } = "";
        public bool Label { get; set; }
    }

    #endregion
}
