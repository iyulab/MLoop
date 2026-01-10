namespace MLoop.Core.Diagnostics;

/// <summary>
/// Scans data directories and identifies unused CSV files.
/// Part of T4.6 - Unused Data Warning feature.
/// </summary>
public class UnusedDataScanner
{
    // Reserved/standard filenames that are expected to be used
    private static readonly HashSet<string> ReservedFileNames = new(StringComparer.OrdinalIgnoreCase)
    {
        "train.csv",
        "validation.csv",
        "test.csv",
        "predict.csv",
        "data.csv"
    };

    /// <summary>
    /// Scans a directory for CSV files and compares against used files.
    /// </summary>
    /// <param name="directory">Directory to scan</param>
    /// <param name="usedFiles">Collection of files that were actually used</param>
    /// <returns>Scan result with unused files</returns>
    public UnusedDataScanResult Scan(string directory, IEnumerable<string> usedFiles)
    {
        var result = new UnusedDataScanResult
        {
            ScannedDirectory = directory
        };

        if (!Directory.Exists(directory))
        {
            result.Error = $"Directory not found: {directory}";
            return result;
        }

        // Get all CSV files in directory
        var allCsvFiles = Directory.GetFiles(directory, "*.csv", SearchOption.TopDirectoryOnly)
            .Select(Path.GetFullPath)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        result.TotalCsvFiles = allCsvFiles.Count;

        // Normalize used files to full paths
        var usedFilesSet = usedFiles
            .Select(f => Path.GetFullPath(f))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        result.UsedFiles = usedFilesSet.ToList();

        // Find unused files
        var unusedFiles = allCsvFiles.Except(usedFilesSet).ToList();

        // Categorize unused files
        foreach (var file in unusedFiles)
        {
            var fileName = Path.GetFileName(file);
            var fileInfo = new FileInfo(file);

            var unusedFile = new UnusedFileInfo
            {
                FullPath = file,
                FileName = fileName,
                SizeBytes = fileInfo.Length,
                LastModified = fileInfo.LastWriteTime
            };

            // Categorize the file
            if (ReservedFileNames.Contains(fileName))
            {
                unusedFile.Category = UnusedFileCategory.ReservedNotUsed;
                unusedFile.Suggestion = "This is a standard file but wasn't used - check if it should be included";
            }
            else if (fileName.StartsWith("backup_", StringComparison.OrdinalIgnoreCase) ||
                     fileName.EndsWith("_backup.csv", StringComparison.OrdinalIgnoreCase) ||
                     fileName.EndsWith(".bak.csv", StringComparison.OrdinalIgnoreCase))
            {
                unusedFile.Category = UnusedFileCategory.Backup;
                unusedFile.Suggestion = "Consider moving backup files to a separate directory";
            }
            else if (fileName.StartsWith("merged_", StringComparison.OrdinalIgnoreCase) ||
                     fileName.Contains("_merged", StringComparison.OrdinalIgnoreCase))
            {
                unusedFile.Category = UnusedFileCategory.MergedOutput;
                unusedFile.Suggestion = "This appears to be a merged output file";
            }
            else if (fileName.StartsWith("temp_", StringComparison.OrdinalIgnoreCase) ||
                     fileName.StartsWith("tmp_", StringComparison.OrdinalIgnoreCase))
            {
                unusedFile.Category = UnusedFileCategory.Temporary;
                unusedFile.Suggestion = "Consider deleting temporary files";
            }
            else
            {
                unusedFile.Category = UnusedFileCategory.Unknown;
                unusedFile.Suggestion = "Could be merged with --auto-merge or specify with --data";
            }

            result.UnusedFiles.Add(unusedFile);
        }

        // Order by size descending for visibility
        result.UnusedFiles = result.UnusedFiles
            .OrderByDescending(f => f.SizeBytes)
            .ToList();

        // Generate summary
        if (result.UnusedFiles.Count == 0)
        {
            result.Summary = "All CSV files in the directory are being used";
        }
        else
        {
            var totalUnusedSize = result.UnusedFiles.Sum(f => f.SizeBytes);
            result.Summary = $"Found {result.UnusedFiles.Count} unused CSV file(s) ({FormatSize(totalUnusedSize)})";

            // Add warnings for significant unused data
            if (result.UnusedFiles.Count > 5)
            {
                result.Warnings.Add($"Large number of unused files ({result.UnusedFiles.Count}) - consider organizing data directory");
            }

            var largeUnused = result.UnusedFiles.Where(f => f.SizeBytes > 10 * 1024 * 1024).ToList();
            if (largeUnused.Count > 0)
            {
                result.Warnings.Add($"{largeUnused.Count} unused file(s) are larger than 10MB");
            }

            // Check for potentially mergeable files
            var potentialMerge = result.UnusedFiles
                .Where(f => f.Category == UnusedFileCategory.Unknown)
                .ToList();

            if (potentialMerge.Count >= 2)
            {
                result.Suggestions.Add($"{potentialMerge.Count} unused files could potentially be merged using --auto-merge");
            }

            // Check for reserved files not used
            var reservedNotUsed = result.UnusedFiles
                .Where(f => f.Category == UnusedFileCategory.ReservedNotUsed)
                .ToList();

            if (reservedNotUsed.Count > 0)
            {
                result.Warnings.Add($"Standard data file(s) not used: {string.Join(", ", reservedNotUsed.Select(f => f.FileName))}");
            }
        }

        return result;
    }

    /// <summary>
    /// Scans a directory and its subdirectories for all CSV files.
    /// </summary>
    /// <param name="directory">Root directory to scan</param>
    /// <param name="usedFiles">Collection of files that were actually used</param>
    /// <param name="recursive">Whether to scan subdirectories</param>
    /// <returns>Comprehensive scan result</returns>
    public UnusedDataScanResult ScanRecursive(string directory, IEnumerable<string> usedFiles, bool recursive = true)
    {
        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var result = new UnusedDataScanResult
        {
            ScannedDirectory = directory
        };

        if (!Directory.Exists(directory))
        {
            result.Error = $"Directory not found: {directory}";
            return result;
        }

        // Get all CSV files
        var allCsvFiles = Directory.GetFiles(directory, "*.csv", searchOption)
            .Select(Path.GetFullPath)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        result.TotalCsvFiles = allCsvFiles.Count;

        // Normalize used files
        var usedFilesSet = usedFiles
            .Select(f => Path.GetFullPath(f))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        result.UsedFiles = usedFilesSet.ToList();

        // Find unused
        foreach (var file in allCsvFiles.Except(usedFilesSet))
        {
            var fileInfo = new FileInfo(file);
            result.UnusedFiles.Add(new UnusedFileInfo
            {
                FullPath = file,
                FileName = Path.GetFileName(file),
                SizeBytes = fileInfo.Length,
                LastModified = fileInfo.LastWriteTime,
                Category = CategorizeFile(Path.GetFileName(file)),
                Suggestion = "Review if this file should be included in training"
            });
        }

        // Generate summary
        if (result.UnusedFiles.Count == 0)
        {
            result.Summary = $"All {result.TotalCsvFiles} CSV files are being used";
        }
        else
        {
            var totalSize = result.UnusedFiles.Sum(f => f.SizeBytes);
            result.Summary = $"Found {result.UnusedFiles.Count}/{result.TotalCsvFiles} unused CSV file(s) ({FormatSize(totalSize)})";
        }

        return result;
    }

    private static UnusedFileCategory CategorizeFile(string fileName)
    {
        if (ReservedFileNames.Contains(fileName))
            return UnusedFileCategory.ReservedNotUsed;

        if (fileName.Contains("backup", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith(".bak.csv", StringComparison.OrdinalIgnoreCase))
            return UnusedFileCategory.Backup;

        if (fileName.Contains("merged", StringComparison.OrdinalIgnoreCase))
            return UnusedFileCategory.MergedOutput;

        if (fileName.StartsWith("temp", StringComparison.OrdinalIgnoreCase) ||
            fileName.StartsWith("tmp", StringComparison.OrdinalIgnoreCase))
            return UnusedFileCategory.Temporary;

        return UnusedFileCategory.Unknown;
    }

    private static string FormatSize(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB" };
        var order = 0;
        double size = bytes;

        while (size >= 1024 && order < sizes.Length - 1)
        {
            order++;
            size /= 1024;
        }

        return $"{size:F1} {sizes[order]}";
    }
}

/// <summary>
/// Result of unused data scan.
/// </summary>
public class UnusedDataScanResult
{
    /// <summary>
    /// Directory that was scanned.
    /// </summary>
    public string ScannedDirectory { get; set; } = "";

    /// <summary>
    /// Total number of CSV files found.
    /// </summary>
    public int TotalCsvFiles { get; set; }

    /// <summary>
    /// Files that were used in training.
    /// </summary>
    public List<string> UsedFiles { get; set; } = new();

    /// <summary>
    /// Files that were not used.
    /// </summary>
    public List<UnusedFileInfo> UnusedFiles { get; set; } = new();

    /// <summary>
    /// Summary of the scan.
    /// </summary>
    public string Summary { get; set; } = "";

    /// <summary>
    /// Warning messages.
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Suggestions for improvement.
    /// </summary>
    public List<string> Suggestions { get; set; } = new();

    /// <summary>
    /// Error message if scan failed.
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Whether there are unused files to report.
    /// </summary>
    public bool HasUnusedFiles => UnusedFiles.Count > 0;
}

/// <summary>
/// Information about an unused file.
/// </summary>
public class UnusedFileInfo
{
    /// <summary>
    /// Full path to the file.
    /// </summary>
    public string FullPath { get; set; } = "";

    /// <summary>
    /// File name only.
    /// </summary>
    public string FileName { get; set; } = "";

    /// <summary>
    /// File size in bytes.
    /// </summary>
    public long SizeBytes { get; set; }

    /// <summary>
    /// Last modification time.
    /// </summary>
    public DateTime LastModified { get; set; }

    /// <summary>
    /// Category of unused file.
    /// </summary>
    public UnusedFileCategory Category { get; set; }

    /// <summary>
    /// Suggestion for handling this file.
    /// </summary>
    public string Suggestion { get; set; } = "";

    /// <summary>
    /// Human-readable file size.
    /// </summary>
    public string SizeFormatted => FormatSize(SizeBytes);

    private static string FormatSize(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB" };
        var order = 0;
        double size = bytes;

        while (size >= 1024 && order < sizes.Length - 1)
        {
            order++;
            size /= 1024;
        }

        return $"{size:F1} {sizes[order]}";
    }
}

/// <summary>
/// Category of unused file.
/// </summary>
public enum UnusedFileCategory
{
    /// <summary>
    /// Unknown category - potentially useful data.
    /// </summary>
    Unknown,

    /// <summary>
    /// Reserved filename (train.csv, test.csv, etc.) that wasn't used.
    /// </summary>
    ReservedNotUsed,

    /// <summary>
    /// Backup file.
    /// </summary>
    Backup,

    /// <summary>
    /// Output from previous merge operation.
    /// </summary>
    MergedOutput,

    /// <summary>
    /// Temporary file.
    /// </summary>
    Temporary
}
