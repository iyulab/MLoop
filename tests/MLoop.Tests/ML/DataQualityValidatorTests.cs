using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class DataQualityValidatorTests : IDisposable
{
    private readonly DataQualityValidator _validator;
    private readonly string _tempDir;

    public DataQualityValidatorTests()
    {
        var mlContext = new MLContext(seed: 42);
        _validator = new DataQualityValidator(mlContext);
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-dqv-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private string CreateCsv(string content)
    {
        var path = Path.Combine(_tempDir, $"data_{Guid.NewGuid():N}.csv");
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }

    #region Constructor

    [Fact]
    public void Constructor_NullMLContext_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new DataQualityValidator(null!));
    }

    #endregion

    #region Empty / Missing Data

    [Fact]
    public void ValidateTrainingData_EmptyFile_ReturnsInvalid()
    {
        var path = CreateCsv("");

        var result = _validator.ValidateTrainingData(path, "Label");

        Assert.False(result.IsValid);
        Assert.Contains("empty", result.ErrorMessage!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidateTrainingData_HeaderOnly_ReturnsInvalid()
    {
        var path = CreateCsv("Feature1,Feature2,Label\n");

        var result = _validator.ValidateTrainingData(path, "Label");

        Assert.False(result.IsValid);
        Assert.Contains("No data rows", result.ErrorMessage!);
    }

    [Fact]
    public void ValidateTrainingData_MissingLabelColumn_ReturnsInvalid()
    {
        var path = CreateCsv("A,B,C\n1,2,3\n4,5,6\n");

        var result = _validator.ValidateTrainingData(path, "NonExistent");

        Assert.False(result.IsValid);
        Assert.Contains("not found", result.ErrorMessage!);
    }

    #endregion

    #region Regression Validation

    [Fact]
    public void ValidateTrainingData_ValidRegression_ReturnsValid()
    {
        var csv = "Feature1,Feature2,Price\n1,2,100\n3,4,200\n5,6,300\n7,8,400\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Price", "regression");

        Assert.True(result.IsValid);
    }

    [Fact]
    public void ValidateTrainingData_AllSameValue_ReturnsInvalid()
    {
        var csv = "Feature1,Feature2,Price\n1,2,0\n3,4,0\n5,6,0\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Price", "regression");

        Assert.False(result.IsValid);
        Assert.Contains("identical", result.ErrorMessage!);
    }

    [Fact]
    public void ValidateTrainingData_NonNumericLabel_Regression_ReturnsInvalid()
    {
        var csv = "Feature1,Label\n1,cat\n2,dog\n3,bird\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Label", "regression");

        Assert.False(result.IsValid);
        Assert.Contains("No valid numeric values", result.ErrorMessage!);
    }

    [Fact]
    public void ValidateTrainingData_LowVariance_ReturnsWarning()
    {
        // All values very close together → low coefficient of variation
        var csv = "Feature1,Price\n1,100.001\n2,100.002\n3,100.003\n4,100.004\n5,100.005\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Price", "regression");

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, w => w.Contains("low variance"));
    }

    #endregion

    #region Classification Validation

    [Fact]
    public void ValidateTrainingData_ValidBinaryClassification_ReturnsValid()
    {
        var csv = "Feature1,Feature2,Target\n1,2,Yes\n3,4,No\n5,6,Yes\n7,8,No\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.True(result.IsValid);
        Assert.Equal(2, result.UniqueClassCount);
    }

    [Fact]
    public void ValidateTrainingData_SingleClass_ReturnsInvalid()
    {
        var csv = "Feature1,Target\n1,Yes\n2,Yes\n3,Yes\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.False(result.IsValid);
        Assert.Contains("only one class", result.ErrorMessage!);
    }

    [Fact]
    public void ValidateTrainingData_TooManyClassesForBinary_ReturnsWarning()
    {
        // Two rows per class: one row per class is now a hard rejection (a class that small cannot
        // be in the train and test partitions at once), which would mask the warning under test.
        var csv = "Feature1,Target\n1,A\n2,A\n3,B\n4,B\n5,C\n6,C\n7,D\n8,D\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, w => w.Contains("classes") && w.Contains("binary"));
    }

    [Fact]
    public void ValidateTrainingData_ClassImbalance_ReturnsWarning()
    {
        // 95:5 ratio → ~19:1 extreme imbalance
        var lines = new List<string> { "Feature1,Target" };
        for (int i = 0; i < 95; i++) lines.Add($"{i},Majority");
        for (int i = 0; i < 5; i++) lines.Add($"{95 + i},Minority");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, w => w.Contains("imbalance"));
        Assert.True(result.ImbalanceRatio >= 10.0);
    }

    [Fact]
    public void ValidateTrainingData_BalancedClasses_NoImbalanceWarning()
    {
        var lines = new List<string> { "Feature1,Target" };
        for (int i = 0; i < 50; i++) lines.Add($"{i},A");
        for (int i = 0; i < 50; i++) lines.Add($"{50 + i},B");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.True(result.IsValid);
        Assert.DoesNotContain(result.Warnings, w => w.Contains("imbalance"));
    }

    #endregion

    #region Multiclass Minimum Samples

    [Fact]
    public void ValidateTrainingData_MulticlassTinyClasses_ReturnsWarning()
    {
        // 4 classes, some with only 2-3 samples → below 5 threshold
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 3; i++) lines.Add($"{i},ClassA");
        for (int i = 0; i < 2; i++) lines.Add($"{10 + i},ClassB");
        for (int i = 0; i < 50; i++) lines.Add($"{20 + i},ClassC");
        for (int i = 0; i < 4; i++) lines.Add($"{70 + i},ClassD");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "multiclass-classification");

        Assert.True(result.IsValid); // warning, not error
        Assert.Contains(result.Warnings, w => w.Contains("Training at risk"));
        Assert.Contains(result.Warnings, w => w.Contains("ClassB")); // 2 samples
    }

    [Fact]
    public void ValidateTrainingData_MulticlassSmallButViable_ReturnsWarning()
    {
        // 3 classes with 8-10 samples each → above 5 but below 15
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 10; i++) lines.Add($"{i},Alpha");
        for (int i = 0; i < 8; i++) lines.Add($"{10 + i},Beta");
        for (int i = 0; i < 9; i++) lines.Add($"{20 + i},Gamma");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "multiclass-classification");

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, w => w.Contains("may be unstable"));
    }

    [Fact]
    public void ValidateTrainingData_MulticlassSufficientSamples_NoWarning()
    {
        // 3 classes with 20+ samples each → above 15 threshold
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 25; i++) lines.Add($"{i},Alpha");
        for (int i = 0; i < 20; i++) lines.Add($"{25 + i},Beta");
        for (int i = 0; i < 22; i++) lines.Add($"{45 + i},Gamma");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "multiclass-classification");

        Assert.True(result.IsValid);
        // Matched on the per-class wording, not on the word "Multiclass" — that word left the
        // messages when the check stopped being multiclass-only, and a substring that can no longer
        // appear would make this assertion pass no matter what the validator did.
        Assert.DoesNotContain(result.Warnings, w => w.Contains("Training at risk") || w.Contains("may be unstable"));
    }

    [Fact]
    public void ValidateTrainingData_BinaryTinyClasses_ReturnsWarning()
    {
        // Binary classification with 3 samples per class. This check used to be multiclass-only,
        // which left extremely imbalanced binary sets — the shape that actually fails in the wild —
        // without any per-class warning. The class count never made it a different problem.
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 3; i++) lines.Add($"{i},Yes");
        for (int i = 0; i < 3; i++) lines.Add($"{10 + i},No");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.True(result.IsValid); // warning, not rejection
        Assert.Contains(result.Warnings, w => w.Contains("Training at risk"));
    }

    [Fact]
    public void ValidateTrainingData_ClassWithSingleSample_IsRejected()
    {
        // One positive among many negatives: a stratified split must keep it in train, so the test
        // partition has no positive and every metric needing one is undefined. Measured end-to-end:
        // without this rejection the run burns its whole time budget and dies inside ML.NET with
        // "AUC is not defined when there is no positive class".
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 40; i++) lines.Add($"{i},No");
        lines.Add("99,Yes");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.False(result.IsValid);
        Assert.Contains("'Yes'", result.ErrorMessage!);
        Assert.Contains("fewer than 2 samples", result.ErrorMessage!);
        // The imbalance advice collected moments earlier (oversample, shorten the time limit) answers
        // a different question and must not compete with the rejection's own explanation.
        Assert.DoesNotContain(result.Suggestions, s => s.Contains("SMOTE"));
    }

    [Fact]
    public void ValidateTrainingData_TwoSampleClass_IsAccepted()
    {
        // The floor is exactly 2: one row can go to train and one to test, so the run completes and
        // reports honest (likely degenerate) metrics for the promotion gate to judge. Rejecting here
        // would refuse work that currently succeeds.
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 40; i++) lines.Add($"{i},No");
        lines.Add("98,Yes");
        lines.Add("99,Yes");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "binary-classification");

        Assert.True(result.IsValid);
    }

    [Fact]
    public void ValidateTrainingData_MulticlassNumericLabels_ChecksMinSamples()
    {
        // Numeric multiclass labels with tiny classes
        var lines = new List<string> { "F1,Target" };
        for (int i = 0; i < 2; i++) lines.Add($"{i},1");
        for (int i = 0; i < 3; i++) lines.Add($"{10 + i},2");
        for (int i = 0; i < 50; i++) lines.Add($"{20 + i},3");
        var path = CreateCsv(string.Join("\n", lines));

        var result = _validator.ValidateTrainingData(path, "Target", "multiclass-classification");

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, w => w.Contains("Training at risk"));
    }

    #endregion

    #region Dataset Size Warnings

    [Fact]
    public void ValidateTrainingData_SmallDataset_ReturnsWarning()
    {
        // 3 features, but only 5 samples (10× rule = need 30)
        var csv = "F1,F2,F3,Label\n1,2,3,10\n4,5,6,20\n7,8,9,30\n10,11,12,40\n13,14,15,50\n";
        var path = CreateCsv(csv);

        var result = _validator.ValidateTrainingData(path, "Label", "regression");

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, w => w.Contains("Small dataset") || w.Contains("Borderline"));
    }

    #endregion
}
