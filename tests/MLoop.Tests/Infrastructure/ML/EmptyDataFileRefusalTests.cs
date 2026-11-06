using System.Text;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// A data file with no content is refused as empty, naming the file. The parser behind the format conversion used to be the first to
/// meet it and printed its own state ("No header record was found", reader and parser offsets) with advice about the label.
/// </summary>
public sealed class EmptyDataFileRefusalTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mloop-emptyfile-" + Guid.NewGuid().ToString("N"));

    public EmptyDataFileRefusalTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string name, string text)
    {
        var path = Path.Combine(_directory, name);
        File.WriteAllText(path, text, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        return path;
    }

    [Fact]
    public void AZeroByteFile_IsRefusedAsEmpty_NamingTheFile()
    {
        var path = Write("members.csv", "");

        var refusal = TrainingInput.EmptyDataFileRefusal(path, "binary-classification");

        Assert.NotNull(refusal);
        Assert.Contains("members.csv", refusal, StringComparison.Ordinal);
        Assert.Contains("empty", refusal, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("\n")]
    [InlineData("\r\n\r\n   \n")]
    [InlineData("\uFEFF\n")]
    public void AFileOfBlankSpaceOnly_IsRefusedAsEmpty(string text)
    {
        var path = Write("blank.csv", text);

        Assert.NotNull(TrainingInput.EmptyDataFileRefusal(path, "regression"));
    }

    [Fact]
    public void AZeroByteFileOfAnyFormat_IsRefusedAsEmpty()
    {
        var path = Write("book.xlsx", "");

        Assert.NotNull(TrainingInput.EmptyDataFileRefusal(path, "regression"));
    }

    [Fact]
    public void AFileWithAHeaderAndNoRows_IsTheHeaderOnlyRefusalsNotThisOne()
    {
        var path = Write("header-only.csv", "a,b,label\n");

        Assert.Null(TrainingInput.EmptyDataFileRefusal(path, "binary-classification"));
    }

    [Fact]
    public void AFileWithContent_IsNotRefused()
    {
        var path = Write("train.csv", "a,b,label\n1,2,yes\n");

        Assert.Null(TrainingInput.EmptyDataFileRefusal(path, "binary-classification"));
    }

    [Fact]
    public void ADirectoryTask_HasNoSingleFileToBeEmpty()
    {
        var path = Write("annotations.json", "");

        Assert.Null(TrainingInput.EmptyDataFileRefusal(path, "image-classification"));
    }

    [Fact]
    public void AMissingFile_IsLeftToTheStepThatSaysItIsMissing()
    {
        Assert.Null(TrainingInput.EmptyDataFileRefusal(Path.Combine(_directory, "nothing.csv"), "regression"));
    }
    // ---- a header and nothing under it ----

    [Theory]
    [InlineData("a,b,label\n")]
    [InlineData("a,b,label")]
    [InlineData("\uFEFFa,b,label\r\n\r\n   \n")]
    [InlineData("\"a\nstill the header\",b,label\n")]
    public void AHeaderAndNoRows_IsRefused_NamingTheFileAndTheCause(string text)
    {
        var path = Write("header-only.csv", text);

        var refusal = TrainingInput.HeaderOnlyDataFileRefusal(path, "binary-classification");

        Assert.NotNull(refusal);
        Assert.Contains("header-only.csv", refusal, StringComparison.Ordinal);
        Assert.Contains("no data rows", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("class", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void ATabSeparatedHeaderAndNoRows_IsRefused()
    {
        var path = Write("header-only.tsv", "a\tb\tlabel\n");

        Assert.NotNull(TrainingInput.HeaderOnlyDataFileRefusal(path, "regression"));
    }

    [Theory]
    [InlineData("a,b,label\n1,2,yes\n")]
    [InlineData("a,b,label\n\n1,2,yes\n")]
    [InlineData("a,b,label\n1,2,yes")]
    [InlineData("a,b,label\n,,\n")]
    public void AFileWithARowUnderItsHeader_IsNotRefused(string text)
    {
        var path = Write("train.csv", text);

        Assert.Null(TrainingInput.HeaderOnlyDataFileRefusal(path, "binary-classification"));
    }

    [Fact]
    public void AFileWithNoHeaderAtAll_IsTheEmptyRefusalsNotThisOne()
    {
        var path = Write("empty.csv", "\n  \n");

        Assert.Null(TrainingInput.HeaderOnlyDataFileRefusal(path, "regression"));
        Assert.NotNull(TrainingInput.EmptyDataFileRefusal(path, "regression"));
    }

    [Fact]
    public void AHeaderOnlyRefusal_LeavesOtherFormatsDirectoryTasksAndMissingFilesAlone()
    {
        var workbook = Write("book.xlsx", "not a real workbook");
        var annotations = Write("annotations.json", "a,b,label\n");

        Assert.Null(TrainingInput.HeaderOnlyDataFileRefusal(workbook, "regression"));
        Assert.Null(TrainingInput.HeaderOnlyDataFileRefusal(annotations, "image-classification"));
        Assert.Null(TrainingInput.HeaderOnlyDataFileRefusal(Path.Combine(_directory, "nothing.csv"), "regression"));
    }
}
