using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Validates schema compatibility between model and prediction data
/// </summary>
public class SchemaValidator
{
    private readonly MLContext _mlContext;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;

    public SchemaValidator(IFileSystemManager fileSystem, IProjectDiscovery projectDiscovery)
    {
        _mlContext = new MLContext(seed: 42);
        _fileSystem = fileSystem;
        _projectDiscovery = projectDiscovery;
    }

    /// <summary>
    /// Validates that the prediction data schema matches the model's expected input schema
    /// </summary>
    /// <param name="modelPath">Path to the model file</param>
    /// <param name="inputDataPath">Path to the prediction data file</param>
    /// <param name="modelName">Model name for loading experiment data</param>
    /// <param name="experimentId">Experiment ID for loading schema</param>
    public async Task<SchemaValidationResult> ValidateAsync(
        string modelPath,
        string inputDataPath,
        string modelName,
        string? experimentId = null)
    {
        var result = new SchemaValidationResult { IsValid = true };

        try
        {
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
                return ValidateWithSavedSchema(savedSchema, inputDataPath);
            }

            // Otherwise, fall back to model-based validation (which is limited)
            var trainedModel = _mlContext.Model.Load(modelPath, out DataViewSchema modelSchema);

            int schemaColumnCount = 0;
            try
            {
                schemaColumnCount = modelSchema.Count;
            }
            catch
            {
                // Ignore if we can't get the count
            }

            if (modelSchema == null || schemaColumnCount == 0)
            {
                // ML.NET models don't store input schema in an accessible way
                // Skip validation and let prediction proceed (it will fail with ML.NET's own error if mismatch)
                result.IsValid = true;  // Allow prediction to proceed
                result.ErrorMessage = "스키마 검증 불가 (저장된 스키마 정보 없음)";
                result.ErrorMessageEn = "Schema validation skipped (no saved schema information)";
                result.Suggestions.Add("Note: This model was trained without schema capture");
                result.Suggestions.Add("Retrain to enable schema validation");
                result.Suggestions.Add("Prediction will proceed - if schema mismatches, ML.NET will report error");
                return result;
            }

            // Load input data to infer schema with UTF-8 encoding
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

            var columns = CsvFieldParser.ParseFields(firstLine);
            var dummyLabel = columns.Length > 0 ? columns[0] : "dummy";

            var columnInference = _mlContext.Auto().InferColumns(
                inputDataPath,
                labelColumnName: dummyLabel,
                separatorChar: ',');

            if (columnInference == null || columnInference.TextLoaderOptions == null)
            {
                result.IsValid = false;
                result.ErrorMessage = "컬럼 정보 추론 실패";
                result.ErrorMessageEn = "Failed to infer column information";
                return result;
            }

            var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var inputData = textLoader.Load(inputDataPath);
            var inputSchema = inputData.Schema;

            // Get model input columns (exclude Label and Features special columns)
            var modelInputColumns = new List<(string Name, DataViewType Type)>();
            foreach (var column in modelSchema)
            {
                // Skip internal ML.NET columns
                if (column.Name == "Label" || column.Name == "Features" ||
                    column.Name == "Score" || column.Name == "PredictedLabel")
                    continue;

                modelInputColumns.Add((column.Name, column.Type));
            }

            // Get input data columns
            var inputColumns = new List<(string Name, DataViewType Type)>();
            foreach (var column in inputSchema)
            {
                inputColumns.Add((column.Name, column.Type));
            }

            // Find missing columns
            var missingColumns = new List<string>();
            var typeMismatchColumns = new List<(string Name, string Expected, string Actual)>();

            foreach (var (modelColName, modelColType) in modelInputColumns)
            {
                var inputCol = inputColumns.FirstOrDefault(c =>
                    c.Name != null && c.Name.Equals(modelColName, StringComparison.OrdinalIgnoreCase));

                if (inputCol.Name == null)
                {
                    // Column might be in input but not matched - check more carefully
                    var exactMatch = inputColumns.Any(c => c.Name != null && c.Name == modelColName);
                    if (!exactMatch)
                    {
                        missingColumns.Add(modelColName);
                    }
                }
                else if (inputCol.Type != null && modelColType != null)
                {
                    // Check type compatibility (simplified - actual check is more complex)
                    if (!AreTypesCompatible(modelColType, inputCol.Type))
                    {
                        typeMismatchColumns.Add((
                            modelColName,
                            modelColType.ToString() ?? "unknown",
                            inputCol.Type.ToString() ?? "unknown"));
                    }
                }
            }

            // Build error message if validation failed
            if (missingColumns.Any() || typeMismatchColumns.Any())
            {
                result.IsValid = false;
                result.MissingColumns = missingColumns;
                result.TypeMismatchColumns = typeMismatchColumns;

                var errorParts = new List<string>();

                if (missingColumns.Any())
                {
                    errorParts.Add($"누락된 컬럼: {string.Join(", ", missingColumns)}");
                    result.ErrorMessageEn = $"Missing columns: {string.Join(", ", missingColumns)}";
                }

                if (typeMismatchColumns.Any())
                {
                    var mismatchDesc = string.Join(", ", typeMismatchColumns.Select(
                        t => $"{t.Name} (expected: {t.Expected}, got: {t.Actual})"));
                    errorParts.Add($"타입 불일치: {mismatchDesc}");

                    if (string.IsNullOrEmpty(result.ErrorMessageEn))
                    {
                        result.ErrorMessageEn = $"Type mismatch: {mismatchDesc}";
                    }
                }

                result.ErrorMessage = string.Join("; ", errorParts);

                // Add helpful suggestions
                result.Suggestions.Add("예측 데이터가 학습 데이터와 동일한 컬럼을 포함하는지 확인하세요.");
                result.Suggestions.Add("Check that prediction data contains the same columns as training data.");
            }

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

    private bool AreTypesCompatible(DataViewType expected, DataViewType actual)
    {
        // For vectors, we need to check dimensions carefully
        // This is simplified - in practice categorical encoding causes dimension issues
        return expected.RawType == actual.RawType;
    }

    /// <summary>
    /// Validates using saved schema information from training
    /// </summary>
    private SchemaValidationResult ValidateWithSavedSchema(InputSchemaInfo savedSchema, string inputDataPath)
    {
        var result = new SchemaValidationResult { IsValid = true };

        try
        {
            // Load input data columns with UTF-8 encoding
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
    public List<(string Name, string Expected, string Actual)> TypeMismatchColumns { get; set; } = new();
    public List<string> Suggestions { get; set; } = new();
}
