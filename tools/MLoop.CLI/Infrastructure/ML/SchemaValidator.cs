using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Validates schema compatibility between model and prediction data
/// </summary>
public class SchemaValidator
{
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;

    public SchemaValidator(IFileSystemManager fileSystem, IProjectDiscovery projectDiscovery)
    {
        _fileSystem = fileSystem;
        _projectDiscovery = projectDiscovery;
    }

    /// <summary>
    /// Checks the prediction data's columns against the input schema recorded when the model was
    /// trained. The model artifact itself is deliberately not consulted — the schema it would expose
    /// describes a featurized view, not the columns a caller supplies.
    /// </summary>
    /// <param name="inputDataPath">Path to the prediction data file</param>
    /// <param name="modelName">Model name for loading experiment data</param>
    /// <param name="experimentId">Experiment ID for loading schema</param>
    public async Task<SchemaValidationResult> ValidateAsync(
        string inputDataPath,
        string modelName,
        string? experimentId = null)
    {
        var result = new SchemaValidationResult { IsValid = true };

        try
        {
            // Read the input through the same encoding detection every other CSV reader uses
            // (CsvDataLoader/CsvHelper/Predict/Train). Forcing UTF-8 here garbles CP949/EUC-KR
            // headers, producing false "missing column" / "not UTF-8" errors on files that
            // train & predict accept.
            var (readPath, _) = EncodingDetector.ConvertToUtf8WithBom(inputDataPath);

            // Try to load saved schema from experiment metadata
            InputSchemaInfo? savedSchema = null;
            if (!string.IsNullOrEmpty(experimentId))
            {
                try
                {
                    var experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
                    var experimentData = await experimentStore.LoadAsync(modelName, experimentId, CancellationToken.None);
                    savedSchema = experimentData?.Config?.InputSchema;
                }
                catch
                {
                    // Continue without saved schema
                }
            }

            // If we have saved schema, use it for validation
            if (savedSchema != null)
            {
                return ValidateWithSavedSchema(savedSchema, readPath);
            }

            // No saved schema: the only honest answer is that nothing was checked.
            //
            // This is reachable only for an experiment whose metadata predates the schema field —
            // training records it unconditionally, on both the tabular and the directory-based path,
            // so nothing this tool produces today arrives here. The alternative once tried, comparing
            // against the model artifact's own schema, cannot work and should not be reintroduced:
            // models are saved without one, and even given one the comparison is against a featurized
            // view where the feature columns have already been folded into a single vector, so present
            // columns read as missing and a label absent by design in prediction data reads as
            // required. Two authorities for the input schema, one of them wrong.
            result.IsValid = true;
            result.ErrorMessage = "스키마 검증 건너뜀 (이 실험에 저장된 입력 스키마가 없습니다).";
            result.ErrorMessageEn = "Schema validation skipped: this experiment has no saved input schema.";
            result.Suggestions.Add("Retrain the model to record the input schema and enable validation.");
            result.Suggestions.Add("Prediction proceeds unchecked — a column mismatch surfaces as a runtime error instead.");
            return result;
        }
        catch (Exception ex)
        {
            result.IsValid = false;
            var innerMsg = ex.InnerException != null ? $" Inner: {ex.InnerException.Message}" : "";
            result.ErrorMessage = $"스키마 검증 중 오류: {ex.Message}{innerMsg}";
            result.ErrorMessageEn = $"Error during schema validation: {ex.Message}{innerMsg}";
            result.Suggestions.Add($"Stack trace: {ex.StackTrace}");
            return result;
        }
    }


    /// <summary>
    /// Validates using saved schema information from training
    /// </summary>
    private SchemaValidationResult ValidateWithSavedSchema(InputSchemaInfo savedSchema, string inputDataPath)
    {
        var result = new SchemaValidationResult { IsValid = true };

        try
        {
            // inputDataPath is already UTF-8 (caller ran EncodingDetector.ConvertToUtf8WithBom)
            string? firstLine;
            using (var reader = new StreamReader(inputDataPath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                result.IsValid = false;
                result.ErrorMessage = "입력 파일이 비어있습니다.";
                result.ErrorMessageEn = "Input file is empty";
                return result;
            }

            var inputColumns = CsvFieldParser.ParseFields(firstLine);
            var missingColumns = new List<string>();
            var extraColumns = new List<string>();
            var potentialEncodingIssues = new List<(string expected, string found)>();

            // Check for missing required columns (Features only, not Label)
            foreach (var savedCol in savedSchema.Columns.Where(c => c.Purpose == "Feature"))
            {
                // Use EXACT match for non-ASCII characters (encoding-sensitive)
                var exactMatch = inputColumns.Any(ic => ic == savedCol.Name);

                if (!exactMatch)
                {
                    // Try case-insensitive for ASCII-only columns
                    var caseInsensitiveMatch = inputColumns.FirstOrDefault(ic =>
                        ic.Equals(savedCol.Name, StringComparison.OrdinalIgnoreCase));

                    if (caseInsensitiveMatch != null)
                    {
                        // ASCII column found with different case - OK
                        continue;
                    }

                    // Check for potential encoding issue (garbled characters)
                    var similarColumn = FindSimilarColumnWithEncodingIssue(savedCol.Name, inputColumns);
                    if (similarColumn != null)
                    {
                        potentialEncodingIssues.Add((savedCol.Name, similarColumn));
                    }

                    missingColumns.Add(savedCol.Name);
                }
            }

            // Check for extra columns (warning only) and identify index columns
            var indexColumns = new List<string>();
            foreach (var inputCol in inputColumns)
            {
                if (!savedSchema.Columns.Any(sc => sc.Name == inputCol ||
                    sc.Name.Equals(inputCol, StringComparison.OrdinalIgnoreCase)))
                {
                    if (MLoop.Core.Data.CsvDataLoader.IsLikelyIndexColumn(inputCol))
                        indexColumns.Add(string.IsNullOrWhiteSpace(inputCol) ? "(empty)" : inputCol);
                    else
                        extraColumns.Add(inputCol);
                }
            }

            // Report results
            if (missingColumns.Any())
            {
                result.IsValid = false;
                result.MissingColumns = missingColumns;

                // Enhanced error message with encoding issue detection
                if (potentialEncodingIssues.Any())
                {
                    result.ErrorMessage = $"필수 컬럼 누락: {string.Join(", ", missingColumns)}\n\n" +
                        "⚠️ 인코딩 이슈 감지:\n" +
                        string.Join("\n", potentialEncodingIssues.Select(p =>
                            $"  기대: '{p.expected}' → 발견: '{p.found}'"));

                    result.ErrorMessageEn = $"Missing required columns: {string.Join(", ", missingColumns)}\n\n" +
                        "⚠️ Encoding issue detected:\n" +
                        string.Join("\n", potentialEncodingIssues.Select(p =>
                            $"  Expected: '{p.expected}' → Found: '{p.found}'"));

                    result.Suggestions.Add("❌ CSV 파일이 UTF-8 인코딩이 아닙니다.");
                    result.Suggestions.Add("✅ 해결방법: 파일을 UTF-8로 변환하거나 UTF-8 BOM을 추가하세요.");
                    result.Suggestions.Add("❌ CSV file is not UTF-8 encoded.");
                    result.Suggestions.Add("✅ Solution: Convert file to UTF-8 or add UTF-8 BOM.");
                }
                else
                {
                    result.ErrorMessage = $"필수 컬럼 누락: {string.Join(", ", missingColumns)}";
                    result.ErrorMessageEn = $"Missing required columns: {string.Join(", ", missingColumns)}";
                    result.Suggestions.Add("확인: 예측 데이터에 학습 시 사용된 모든 Feature 컬럼이 포함되어 있는지 확인하세요");
                    result.Suggestions.Add("Check: Ensure prediction data contains all Feature columns used during training");
                }
            }
            else if (indexColumns.Any() || extraColumns.Any())
            {
                // Extra columns are ok, just warn
                result.IsValid = true;
                if (indexColumns.Any())
                {
                    result.Suggestions.Add($"참고: 인덱스 컬럼 감지됨 (자동 제거): {string.Join(", ", indexColumns)}");
                    result.Suggestions.Add($"Note: Index column(s) detected (auto-removed): {string.Join(", ", indexColumns)}");
                    result.Suggestions.Add("💡 pandas에서 CSV 저장 시 index=False 옵션 사용을 권장합니다.");
                }
                if (extraColumns.Any())
                {
                    result.Suggestions.Add($"참고: 추가 컬럼 발견 (무시됨): {string.Join(", ", extraColumns)}");
                    result.Suggestions.Add($"Note: Extra columns found (will be ignored): {string.Join(", ", extraColumns)}");
                }
            }

            return result;
        }
        catch (Exception ex)
        {
            result.IsValid = false;
            result.ErrorMessage = $"스키마 검증 중 오류: {ex.Message}";
            result.ErrorMessageEn = $"Error during schema validation: {ex.Message}";
            return result;
        }
    }

    /// <summary>
    /// Find similar column name that might have encoding issues
    /// </summary>
    private static string? FindSimilarColumnWithEncodingIssue(string expectedName, string[] inputColumns)
    {
        // Check if expected name contains non-ASCII characters (Korean, Japanese, Chinese, etc.)
        bool hasNonAscii = expectedName.Any(c => c > 127);

        if (!hasNonAscii)
            return null; // ASCII columns don't have encoding issues

        // Look for columns with similar length (±2 chars) that might be garbled
        foreach (var inputCol in inputColumns)
        {
            int lengthDiff = Math.Abs(inputCol.Length - expectedName.Length);

            // Garbled UTF-8 text often has similar length
            if (lengthDiff <= 2)
            {
                // Check if input column has replacement characters or garbled bytes
                if (inputCol.Contains('\uFFFD') || // Replacement character
                    inputCol.Any(c => c > 127 && c < 256)) // Latin-1/CP949 range
                {
                    return inputCol;
                }
            }
        }

        return null;
    }
}

/// <summary>
/// Result of schema validation
/// </summary>
public class SchemaValidationResult
{
    public bool IsValid { get; set; }
    public string? ErrorMessage { get; set; }
    public string? ErrorMessageEn { get; set; }
    public List<string> MissingColumns { get; set; } = new();
    public List<string> Suggestions { get; set; } = new();
}
