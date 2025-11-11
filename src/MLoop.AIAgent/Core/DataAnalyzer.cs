using CsvHelper;
using CsvHelper.Configuration;
using MLoop.AIAgent.Core.Models;
using System.Globalization;
using System.Text.Json;

namespace MLoop.AIAgent.Core;

/// <summary>
/// Analyzes CSV/JSON datasets for ML readiness
/// </summary>
public class DataAnalyzer
{
    private const int MaxSampleSize = 10000; // Limit for large datasets
    private const int SampleValuesCount = 5; // Number of sample values to show

    /// <summary>
    /// Analyze a CSV or JSON file
    /// </summary>
    public async Task<DataAnalysisReport> AnalyzeAsync(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}");
        }

        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        
        return extension switch
        {
            ".csv" => await AnalyzeCsvAsync(filePath),
            ".json" or ".jsonl" => await AnalyzeJsonAsync(filePath),
            _ => throw new NotSupportedException($"File type not supported: {extension}")
        };
    }

    private async Task<DataAnalysisReport> AnalyzeCsvAsync(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        var records = new List<Dictionary<string, string>>();

        // Read CSV with CsvHelper
        using (var reader = new StreamReader(filePath))
        using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HeaderValidated = null,
            MissingFieldFound = null
        }))
        {
            await csv.ReadAsync();
            csv.ReadHeader();
            var headers = csv.HeaderRecord ?? Array.Empty<string>();

            int rowCount = 0;
            while (await csv.ReadAsync() && rowCount < MaxSampleSize)
            {
                var record = new Dictionary<string, string>();
                foreach (var header in headers)
                {
                    record[header] = csv.GetField(header) ?? string.Empty;
                }
                records.Add(record);
                rowCount++;
            }
        }

        return BuildReport(filePath, fileInfo.Length, records);
    }

    private async Task<DataAnalysisReport> AnalyzeJsonAsync(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        var content = await File.ReadAllTextAsync(filePath);
        
        // Try to parse as JSON array
        List<Dictionary<string, string>> records;
        
        try
        {
            var jsonArray = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(content);
            if (jsonArray == null)
            {
                throw new InvalidDataException("Invalid JSON format");
            }

            records = jsonArray
                .Take(MaxSampleSize)
                .Select(dict => dict.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value.ToString()))
                .ToList();
        }
        catch
        {
            // Try JSONL format (one JSON object per line)
            var lines = content.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Take(MaxSampleSize);
            
            records = lines
                .Select(line =>
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(line);
                    return dict?.ToDictionary(
                        kvp => kvp.Key,
                        kvp => kvp.Value.ToString()) ?? new Dictionary<string, string>();
                })
                .ToList();
        }

        return BuildReport(filePath, fileInfo.Length, records);
    }

    private DataAnalysisReport BuildReport(
        string filePath, 
        long fileSize, 
        List<Dictionary<string, string>> records)
    {
        if (records.Count == 0)
        {
            throw new InvalidDataException("No data records found");
        }

        var columns = AnalyzeColumns(records);
        var qualityIssues = DetectQualityIssues(columns, records);
        var targetRecommendation = RecommendTarget(columns);
        var mlReadiness = AssessMLReadiness(columns, qualityIssues);

        return new DataAnalysisReport
        {
            FilePath = filePath,
            RowCount = records.Count,
            ColumnCount = columns.Count,
            FileSizeBytes = fileSize,
            Columns = columns,
            RecommendedTarget = targetRecommendation,
            QualityIssues = qualityIssues,
            MLReadiness = mlReadiness
        };
    }

    private List<ColumnAnalysis> AnalyzeColumns(List<Dictionary<string, string>> records)
    {
        var columns = new List<ColumnAnalysis>();
        var headers = records[0].Keys;

        foreach (var header in headers)
        {
            var values = records.Select(r => r.TryGetValue(header, out var v) ? v : string.Empty).ToList();
            
            var nonNullValues = values.Where(v => !string.IsNullOrWhiteSpace(v)).ToList();
            var nullCount = values.Count - nonNullValues.Count;
            
            var inferredType = InferDataType(nonNullValues);
            var uniqueCount = nonNullValues.Distinct().Count();



            // Add type-specific statistics
            NumericStatistics? numericStats = null;
            CategoricalStatistics? categoricalStats = null;

            if (inferredType == DataType.Numeric)
            {
                numericStats = CalculateNumericStats(nonNullValues);
            }
            else if (inferredType == DataType.Categorical || inferredType == DataType.Boolean)
            {
                categoricalStats = CalculateCategoricalStats(nonNullValues);
            }

            var column = new ColumnAnalysis
            {
                Name = header,
                InferredType = inferredType,
                NonNullCount = nonNullValues.Count,
                NullCount = nullCount,
                UniqueCount = uniqueCount,
                SampleValues = nonNullValues.Take(SampleValuesCount).ToList(),
                NumericStats = numericStats,
                CategoricalStats = categoricalStats
            };

            columns.Add(column);
        }

        return columns;
    }

    private DataType InferDataType(List<string> values)
    {
        if (values.Count == 0) return DataType.Unknown;

        var sample = values.Take(100).ToList();

        // Check if numeric
        var numericCount = sample.Count(v => double.TryParse(v, out _));
        if (numericCount > sample.Count * 0.8)
        {
            return DataType.Numeric;
        }

        // Check if datetime
        var dateCount = sample.Count(v => DateTime.TryParse(v, out _));
        if (dateCount > sample.Count * 0.8)
        {
            return DataType.DateTime;
        }

        // Check if boolean
        var booleanValues = new HashSet<string>(StringComparer.OrdinalIgnoreCase) 
            { "true", "false", "yes", "no", "1", "0", "y", "n" };
        var distinctValues = sample.Select(v => v.ToLowerInvariant()).Distinct().ToList();
        if (distinctValues.Count == 2 && distinctValues.All(v => booleanValues.Contains(v)))
        {
            return DataType.Boolean;
        }

        // Check if categorical (low cardinality) or text (high cardinality)
        var uniqueCount = values.Distinct().Count();
        var cardinalityRatio = uniqueCount / (double)values.Count;

        if (cardinalityRatio < 0.1 || uniqueCount < 50)
        {
            return DataType.Categorical;
        }

        return DataType.Text;
    }

    private NumericStatistics CalculateNumericStats(List<string> values)
    {
        var numbers = values
            .Where(v => double.TryParse(v, out _))
            .Select(double.Parse)
            .OrderBy(n => n)
            .ToList();

        if (numbers.Count == 0)
        {
            return new NumericStatistics
            {
                Mean = 0, Median = 0, StandardDeviation = 0, Variance = 0,
                Min = 0, Max = 0, Q1 = 0, Q3 = 0
            };
        }

        var mean = numbers.Average();
        var variance = numbers.Sum(n => Math.Pow(n - mean, 2)) / numbers.Count;
        var stdDev = Math.Sqrt(variance);

        var median = CalculatePercentile(numbers, 0.5);
        var q1 = CalculatePercentile(numbers, 0.25);
        var q3 = CalculatePercentile(numbers, 0.75);
        var iqr = q3 - q1;

        // Detect outliers using IQR method
        var lowerBound = q1 - 1.5 * iqr;
        var upperBound = q3 + 1.5 * iqr;
        var outliers = numbers.Where(n => n < lowerBound || n > upperBound).ToList();

        return new NumericStatistics
        {
            Mean = mean,
            Median = median,
            StandardDeviation = stdDev,
            Variance = variance,
            Min = numbers.First(),
            Max = numbers.Last(),
            Q1 = q1,
            Q3 = q3,
            OutlierCount = outliers.Count,
            OutlierValues = outliers.Take(10).ToList() // Limit outlier samples
        };
    }

    private CategoricalStatistics CalculateCategoricalStats(List<string> values)
    {
        var valueCounts = values
            .GroupBy(v => v)
            .ToDictionary(g => g.Key, g => g.Count())
            .OrderByDescending(kvp => kvp.Value)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        var mostFrequent = valueCounts.FirstOrDefault();

        return new CategoricalStatistics
        {
            MostFrequentValue = mostFrequent.Key ?? string.Empty,
            MostFrequentCount = mostFrequent.Value,
            ValueCounts = valueCounts
        };
    }

    private double CalculatePercentile(List<double> sortedNumbers, double percentile)
    {
        if (sortedNumbers.Count == 0) return 0;
        
        var index = percentile * (sortedNumbers.Count - 1);
        var lower = (int)Math.Floor(index);
        var upper = (int)Math.Ceiling(index);
        
        if (lower == upper) return sortedNumbers[lower];
        
        var weight = index - lower;
        return sortedNumbers[lower] * (1 - weight) + sortedNumbers[upper] * weight;
    }

    private DataQualityIssues DetectQualityIssues(
        List<ColumnAnalysis> columns, 
        List<Dictionary<string, string>> records)
    {
        var issues = new DataQualityIssues
        {
            ColumnsWithMissingValues = columns
                .Where(c => c.MissingPercentage > 5) // >5% missing
                .Select(c => c.Name)
                .ToList(),

            ColumnsWithOutliers = columns
                .Where(c => c.NumericStats != null && c.NumericStats.OutlierCount > 0)
                .Select(c => c.Name)
                .ToList(),

            HighCardinalityColumns = columns
                .Where(c => c.CategoricalStats != null && c.CategoricalStats.IsHighCardinality)
                .Select(c => c.Name)
                .ToList(),

            ConstantColumns = columns
                .Where(c => c.UniqueCount == 1)
                .Select(c => c.Name)
                .ToList(),

            DuplicateRowCount = DetectDuplicateRows(records)
        };

        return issues;
    }

    private int DetectDuplicateRows(List<Dictionary<string, string>> records)
    {
        var uniqueRows = new HashSet<string>();
        int duplicates = 0;

        foreach (var record in records)
        {
            var rowKey = string.Join("|", record.Values);
            if (!uniqueRows.Add(rowKey))
            {
                duplicates++;
            }
        }

        return duplicates;
    }

    private TargetRecommendation? RecommendTarget(List<ColumnAnalysis> columns)
    {
        // Prefer columns with low cardinality for classification
        var candidates = columns
            .Where(c => c.InferredType == DataType.Categorical || 
                       c.InferredType == DataType.Boolean ||
                       c.InferredType == DataType.Numeric)
            .OrderBy(c => c.MissingPercentage)
            .ToList();

        if (candidates.Count == 0) return null;

        foreach (var candidate in candidates)
        {
            // Binary classification
            if (candidate.UniqueCount == 2)
            {
                return new TargetRecommendation
                {
                    ColumnName = candidate.Name,
                    ProblemType = MLProblemType.BinaryClassification,
                    Confidence = 0.8,
                    Reason = $"Column has exactly 2 unique values, suitable for binary classification",
                    AlternativeTargets = candidates.Take(3).Select(c => c.Name).ToList()
                };
            }

            // Multiclass classification
            if (candidate.InferredType == DataType.Categorical && 
                candidate.UniqueCount > 2 && 
                candidate.UniqueCount < 20)
            {
                return new TargetRecommendation
                {
                    ColumnName = candidate.Name,
                    ProblemType = MLProblemType.MulticlassClassification,
                    Confidence = 0.7,
                    Reason = $"Column has {candidate.UniqueCount} unique values, suitable for multiclass classification",
                    AlternativeTargets = candidates.Take(3).Select(c => c.Name).ToList()
                };
            }

            // Regression
            if (candidate.InferredType == DataType.Numeric)
            {
                return new TargetRecommendation
                {
                    ColumnName = candidate.Name,
                    ProblemType = MLProblemType.Regression,
                    Confidence = 0.6,
                    Reason = "Column is numeric and continuous, suitable for regression",
                    AlternativeTargets = candidates.Take(3).Select(c => c.Name).ToList()
                };
            }
        }

        return null;
    }

    private MLReadinessAssessment AssessMLReadiness(
        List<ColumnAnalysis> columns, 
        DataQualityIssues issues)
    {
        var blockingIssues = new List<string>();
        var warnings = new List<string>();
        var recommendations = new List<string>();

        // Check for blocking issues
        if (columns.Count < 2)
        {
            blockingIssues.Add("Dataset must have at least 2 columns (features + target)");
        }

        if (issues.ConstantColumns.Count > 0)
        {
            warnings.Add($"{issues.ConstantColumns.Count} constant columns detected - consider removing them");
        }

        if (issues.ColumnsWithMissingValues.Count > 0)
        {
            warnings.Add($"{issues.ColumnsWithMissingValues.Count} columns have >5% missing values");
            recommendations.Add("Consider using preprocessing scripts to handle missing values");
        }

        if (issues.ColumnsWithOutliers.Count > 0)
        {
            warnings.Add($"{issues.ColumnsWithOutliers.Count} columns have outliers detected");
            recommendations.Add("Review outliers - may need capping or transformation");
        }

        if (issues.HighCardinalityColumns.Count > 0)
        {
            warnings.Add($"{issues.HighCardinalityColumns.Count} high-cardinality categorical columns");
            recommendations.Add("High-cardinality columns may need encoding or feature engineering");
        }

        if (issues.DuplicateRowCount > 0)
        {
            warnings.Add($"{issues.DuplicateRowCount} duplicate rows detected");
            recommendations.Add("Consider removing duplicate rows before training");
        }

        // Calculate readiness score
        double score = 1.0;
        score -= blockingIssues.Count * 0.5;
        score -= warnings.Count * 0.1;
        score = Math.Max(0, Math.Min(1, score));

        return new MLReadinessAssessment
        {
            IsReady = blockingIssues.Count == 0,
            ReadinessScore = score,
            BlockingIssues = blockingIssues,
            Warnings = warnings,
            Recommendations = recommendations
        };
    }
}
