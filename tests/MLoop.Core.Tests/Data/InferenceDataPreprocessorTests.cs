using System.Text;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Unit tests for <see cref="InferenceDataPreprocessor"/> — the single, shared CSV preprocessing
/// sequence that predict and evaluate apply at inference time. These tests pin the convergence that
/// resolved BUG-43/44: encoding (CP949), multiline flattening, index removal, and schema-driven /
/// fallback column exclusion must all match what training did, so the model's feature width lines up.
/// </summary>
public class InferenceDataPreprocessorTests : IDisposable
{
    private readonly string _testDir;

    public InferenceDataPreprocessorTests()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-prep-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Prepare_Cp949WithKoreanHeader_PreservesKoreanColumns()
    {
        // BUG-43 class for predict: a CP949 file read as UTF-8 garbles the Korean header.
        var path = Path.Combine(_testDir, "cp949.csv");
        File.WriteAllText(path, "입력값,출력값\n1,2\n3,4\n", Encoding.GetEncoding(949));

        var result = InferenceDataPreprocessor.Prepare(path, "출력값", trainedSchema: null, out var tempFiles);

        var header = ReadHeader(result);
        Assert.Contains("입력값", header);
        Assert.Contains("출력값", header);
        CleanupTemps(tempFiles);
    }

    [Fact]
    public void Prepare_MultiLineQuotedField_FlattensToSingleRow()
    {
        // Flatten was missing from both inference paths — a multiline quoted field made TextLoader
        // see extra physical rows, breaking column/row alignment.
        var path = CreateBomCsv("multiline.csv",
            "Id,Comment,Label\n1,\"line one\nline two\",A\n2,single,B\n");

        var result = InferenceDataPreprocessor.Prepare(path, "Label", trainedSchema: null, out var tempFiles);

        var lines = File.ReadAllLines(result);
        // header + 2 data rows (the multiline field collapsed into one physical line)
        Assert.Equal(3, lines.Length);
        CleanupTemps(tempFiles);
    }

    [Fact]
    public void Prepare_UnnamedIndexColumn_IsRemoved()
    {
        var path = CreateBomCsv("indexed.csv", "Unnamed: 0,Feature1,Label\n0,1.5,A\n1,2.5,B\n");

        var result = InferenceDataPreprocessor.Prepare(path, "Label", trainedSchema: null, out var tempFiles);

        var header = ReadHeader(result);
        Assert.DoesNotContain("Unnamed: 0", header);
        Assert.Contains("Feature1", header);
        CleanupTemps(tempFiles);
    }

    [Fact]
    public void Prepare_WithSchema_RemovesExcludedColumns()
    {
        // Schema-driven exclusion: training marked "Created" as Exclude (a DateTime column).
        var path = CreateBomCsv("excluded.csv",
            "Feature1,Created,Label\n1.5,2024-01-01,A\n2.5,2024-01-02,B\n");
        var schema = SchemaWith(
            ("Feature1", "Numeric", "Feature"),
            ("Created", "Text", "Exclude"),
            ("Label", "Categorical", "Label"));

        var result = InferenceDataPreprocessor.Prepare(path, "Label", schema, out var tempFiles);

        var header = ReadHeader(result);
        Assert.DoesNotContain("Created", header);
        Assert.Contains("Feature1", header);
        Assert.Contains("Label", header);
        CleanupTemps(tempFiles);
    }

    [Fact]
    public void Prepare_NoSchema_RemovesConstantColumns()
    {
        // Constant-column removal was present in train and missing from predict's no-schema fallback.
        // Without it, a constant column survives into inference and widens the feature vector.
        var path = CreateBomCsv("constant.csv",
            "Feature1,Const,Label\n1.5,X,A\n2.5,X,B\n3.5,X,A\n");

        var result = InferenceDataPreprocessor.Prepare(path, "Label", trainedSchema: null, out var tempFiles);

        var header = ReadHeader(result);
        Assert.DoesNotContain("Const", header);
        Assert.Contains("Feature1", header);
        CleanupTemps(tempFiles);
    }

    [Fact]
    public void Prepare_PredictAndEvaluatePathsProduceSameColumns()
    {
        // The whole point of the component: a single sequence means both inference callers derive an
        // identical feature set from identical input. Calling Prepare twice models the two callers.
        var contents = "Unnamed: 0,Feature1,Const,Label\n0,1.5,X,A\n1,2.5,X,B\n2,3.5,X,A\n";
        var pathA = CreateBomCsv("pathA.csv", contents);
        var pathB = CreateBomCsv("pathB.csv", contents);

        var resultA = InferenceDataPreprocessor.Prepare(pathA, "Label", trainedSchema: null, out var tempsA);
        var resultB = InferenceDataPreprocessor.Prepare(pathB, "Label", trainedSchema: null, out var tempsB);

        Assert.Equal(ReadHeader(resultA), ReadHeader(resultB));
        CleanupTemps(tempsA);
        CleanupTemps(tempsB);
    }

    [Fact]
    public void Prepare_TempFilesTracked_NoOrphansAfterCleanup()
    {
        // Every temp the sequence creates must be reported so the caller can clean up after lazy
        // consumption. After cleanup nothing the component created may remain on disk.
        var path = CreateBomCsv("orphan.csv", "Unnamed: 0,Feature1,Label\n0,1.5,A\n1,2.5,B\n");

        var result = InferenceDataPreprocessor.Prepare(path, "Label", trainedSchema: null, out var tempFiles);

        // The result file is among the reported temps (index removal created a new file).
        Assert.Contains(result, tempFiles);
        CleanupTemps(tempFiles);
        foreach (var t in tempFiles)
            Assert.False(File.Exists(t), $"temp not cleaned: {t}");
        // Original input is untouched.
        Assert.True(File.Exists(path));
    }

    [Fact]
    public void Prepare_NoChangesNeeded_ReturnsInputAndReportsNoTemps()
    {
        // A clean UTF-8 BOM file with no index/constant/multiline columns needs no rewriting.
        var path = CreateBomCsv("clean.csv", "Feature1,Feature2,Label\n1.5,9.0,A\n2.5,8.0,B\n3.5,7.0,A\n");

        var result = InferenceDataPreprocessor.Prepare(path, "Label", trainedSchema: null, out var tempFiles);

        Assert.Equal(path, result);
        Assert.Empty(tempFiles);
    }

    #region Helpers

    private string CreateBomCsv(string fileName, string content)
    {
        var path = Path.Combine(_testDir, fileName);
        File.WriteAllText(path, content, new UTF8Encoding(true));
        return path;
    }

    private static string[] ReadHeader(string path)
    {
        using var reader = new StreamReader(path, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        var line = reader.ReadLine() ?? string.Empty;
        return line.Split(',').Select(s => s.Trim().Trim('"')).ToArray();
    }

    private static InputSchemaInfo SchemaWith(params (string Name, string DataType, string Purpose)[] cols)
    {
        return new InputSchemaInfo
        {
            Columns = cols.Select(c => new ColumnSchema
            {
                Name = c.Name,
                DataType = c.DataType,
                Purpose = c.Purpose
            }).ToList(),
            CapturedAt = new DateTime(2026, 1, 1)
        };
    }

    private static void CleanupTemps(List<string> tempFiles)
    {
        foreach (var t in tempFiles)
        {
            if (File.Exists(t))
            {
                try { File.Delete(t); } catch { }
            }
        }
    }

    #endregion
}
