using System.Globalization;
using System.IO.Compression;
using System.Text.RegularExpressions;
using FilePrepper.Utils;
using MLoop.Core.Storage;

namespace MLoop.API;

/// <summary>What the server takes in one upload, how big a zip may grow when it is unpacked, and how much the stored uploads may
/// add up to — <c>Serve:Datasets</c> in the server's configuration. The defaults are ceilings for a project on one machine, not
/// measured thresholds; a deployment that takes larger datasets raises them.</summary>
public sealed record DatasetLimits(long MaxUploadBytes, int MaxEntries, long MaxExpandedBytes, long MaxTotalBytes = DatasetLimits.DefaultMaxTotalBytes)
{
    public const long DefaultMaxUploadBytes = 2L << 30;
    public const int DefaultMaxEntries = 100_000;
    public const long DefaultMaxExpandedBytes = 8L << 30;
    public const long DefaultMaxTotalBytes = 32L << 30;

    public static DatasetLimits From(IConfiguration configuration) => new(
        configuration.GetValue("Serve:Datasets:MaxUploadBytes", DefaultMaxUploadBytes),
        configuration.GetValue("Serve:Datasets:MaxEntries", DefaultMaxEntries),
        configuration.GetValue("Serve:Datasets:MaxExpandedBytes", DefaultMaxExpandedBytes),
        configuration.GetValue("Serve:Datasets:MaxTotalBytes", DefaultMaxTotalBytes));
}

/// <summary>An upload the server will not take, and why — the caller's mistake (400), or a limit (413).</summary>
public sealed class DatasetRefusedException(string message, int statusCode = StatusCodes.Status400BadRequest) : Exception(message)
{
    public int StatusCode { get; } = statusCode;
}

/// <summary>What a retention pass did: uploads removed (and their bytes), uploads kept because a job still reads them, and the cutoff used.</summary>
public sealed record DatasetPruneReport(int Removed, int KeptForJobs, long Bytes, DateTime Cutoff);

/// <summary>One stored upload: where it is in the project (relative, <c>/</c>-separated — what <c>POST /train</c> takes as <c>dataFile</c>) and what it holds.</summary>
public sealed record StoredDataset(string Id, string Kind, string Path, int Files, long Bytes, DateTime CreatedAt);

/// <summary>
/// The training data a client sent to the server (<c>POST /datasets</c>): a table file, or a zip of a directory dataset (class
/// subfolders of images; a COCO annotations file with its images). It becomes files under the project's
/// <c>datasets/uploads/&lt;id&gt;/</c> — the place <c>POST /train</c> and <c>mloop train</c> already read from — so an upload is
/// not a second kind of data, only a way for it to get there.
/// </summary>
/// <remarks>
/// This is the one place the server writes bytes a client chose, so every rule of what may land on the disk is here: the
/// size of the upload and of what a zip unpacks to are counted as the bytes arrive (a zip's declared sizes are the sender's
/// word), an entry's name must stay inside the upload's folder (no absolute path, drive, <c>..</c> segment or backslash),
/// a link or any non-regular entry is refused, and so is a file whose extension is not one a dataset holds. An upload is
/// written aside and moved into place whole, so a refused or failed one leaves nothing, and a reader never sees half of one.
/// </remarks>
public sealed class DatasetStore(string projectRoot)
{
    /// <summary>The folder under the project's <c>datasets/</c> where uploads live.</summary>
    public const string UploadsDirectoryName = "uploads";

    private static readonly Regex IdPattern = new(@"^ds-\d{8}-\d{6}-[0-9a-f]{8}$", RegexOptions.Compiled);

    /// <summary>What a directory dataset holds: images (what the image loaders read), COCO annotations, YOLO labels and class lists, tables.</summary>
    private static readonly HashSet<string> ZipFileExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".json", ".txt", ".names", ".yaml", ".yml", ".csv", ".tsv",
    };

    private string UploadsDirectory => Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory, UploadsDirectoryName);

    /// <summary>Whether <paramref name="id"/> is an id this store makes — the only shape a request's id is let into a path as.</summary>
    public static bool IsId(string id) => IdPattern.IsMatch(id);

    /// <summary>Keeps <paramref name="content"/> as a dataset. <paramref name="check"/> judges the dataset's place before it is
    /// kept (the training layout of a task, say): a text refuses it.</summary>
    /// <remarks>
    /// <paramref name="declaredLength"/> is the length the sender announced, when it did: an upload that cannot fit is refused
    /// before a byte is read. What the stored uploads add up to is judged against <see cref="DatasetLimits.MaxTotalBytes"/> as
    /// the upload begins and counted as it arrives — a ceiling, not a quota: two uploads in flight at once are each judged
    /// against what was stored when they began.
    /// </remarks>
    /// <exception cref="DatasetRefusedException">The upload cannot be taken — nothing is left of it.</exception>
    public async Task<StoredDataset> SaveAsync(
        Stream content, string? fileName, DatasetLimits limits, Func<string, string?>? check, CancellationToken ct,
        long? declaredLength = null)
    {
        var name = CheckFileName(fileName);
        var isZip = name.EndsWith(".zip", StringComparison.OrdinalIgnoreCase);
        if (!isZip && !DataFileFormats.IsReadable(name))
            throw new DatasetRefusedException(
                $"'{name}' is neither a table file the trainers read (csv, tsv, xlsx, parquet, jsonl …) nor a .zip of a directory dataset.");

        // What the disk may still take: the most one upload and one unpacked zip may then be is that, and the refusal names the setting that bound it.
        var room = limits.MaxTotalBytes - StoredBytes();
        var totalSetting = $"Serve:Datasets:MaxTotalBytes — the stored uploads would pass {limits.MaxTotalBytes.ToString("N0", CultureInfo.InvariantCulture)} bytes; DELETE /datasets/{{id}} or Serve:Datasets:RetentionDays frees room";
        if (room <= 0 || declaredLength > room)
            throw new DatasetRefusedException(
                $"The upload does not fit: the stored uploads leave {Math.Max(room, 0).ToString("N0", CultureInfo.InvariantCulture)} bytes of room ({totalSetting}).",
                StatusCodes.Status413PayloadTooLarge);
        var uploadLimit = Math.Min(limits.MaxUploadBytes, room);
        var uploadSetting = room < limits.MaxUploadBytes ? totalSetting : "Serve:Datasets:MaxUploadBytes";
        var expandedLimit = Math.Min(limits.MaxExpandedBytes, room);
        var expandedSetting = room < limits.MaxExpandedBytes ? totalSetting : "Serve:Datasets:MaxExpandedBytes";

        var createdAt = DateTime.UtcNow;
        var id = $"ds-{TimestampKeys.Stamp(createdAt)}-{Guid.NewGuid().ToString("N")[..8]}";
        var uploads = UploadsDirectory;
        Directory.CreateDirectory(uploads);
        var staging = Path.Combine(uploads, $".incoming-{id}");
        var final = Path.Combine(uploads, id);
        Directory.CreateDirectory(staging);
        try
        {
            string datasetPath;
            if (isZip)
            {
                var archive = Path.Combine(staging, ".upload.zip");
                await using (var file = File.Create(archive))
                    await CopyLimitedAsync(content, file, uploadLimit, "The upload", uploadSetting, ct);
                var unpacked = Path.Combine(staging, "content");
                Directory.CreateDirectory(unpacked);
                await UnpackAsync(archive, unpacked, limits, expandedLimit, expandedSetting, ct);
                File.Delete(archive);
                // A zip of «the folder» holds that one folder at its root: it is the dataset, as it would be on disk.
                var children = Directory.GetFileSystemEntries(unpacked);
                datasetPath = children.Length == 1 && Directory.Exists(children[0]) ? children[0] : unpacked;
                if (Directory.GetFileSystemEntries(datasetPath).Length == 0)
                    throw new DatasetRefusedException("The zip holds no files.");
            }
            else
            {
                datasetPath = Path.Combine(staging, name);
                await using (var file = File.Create(datasetPath))
                {
                    if (await CopyLimitedAsync(content, file, uploadLimit, "The upload", uploadSetting, ct) == 0)
                        throw new DatasetRefusedException("The request body is empty: send the file's bytes as the body.");
                }
            }

            if (check?.Invoke(datasetPath) is { } refusal)
                throw new DatasetRefusedException(refusal);

            var (files, bytes) = Count(datasetPath);
            // Where it will be once moved: the same path from the upload's folder.
            var relative = isZip
                ? Path.GetRelativePath(staging, datasetPath).Replace(Path.DirectorySeparatorChar, '/')
                : name;
            Directory.Move(staging, final);
            var projectRelative = $"{ProjectLayout.DatasetsDirectory}/{UploadsDirectoryName}/{id}/{relative}";
            return new StoredDataset(id, isZip ? "directory" : "table", projectRelative, files, bytes, createdAt);
        }
        catch
        {
            TryDelete(staging);
            throw;
        }
    }

    /// <summary>What the stored uploads add up to, in bytes.</summary>
    public long StoredBytes() => List().Sum(d => d.Bytes);

    /// <summary>The stored uploads, newest first.</summary>
    public IReadOnlyList<StoredDataset> List()
    {
        if (!Directory.Exists(UploadsDirectory))
            return [];
        return [.. Directory.EnumerateDirectories(UploadsDirectory)
            .Where(d => IsId(Path.GetFileName(d)))
            .Select(Describe)
            .OfType<StoredDataset>()
            .OrderByDescending(d => d.Id, StringComparer.Ordinal)];
    }

    /// <summary>One stored upload, or <c>null</c> when there is none by that id.</summary>
    public StoredDataset? Find(string id) => IsId(id) && Directory.Exists(Path.Combine(UploadsDirectory, id))
        ? Describe(Path.Combine(UploadsDirectory, id))
        : null;

    /// <summary>The upload's folder (for a caller that asks whether a job still reads it), or <c>null</c> for an id that is not one.</summary>
    public string? FolderOf(string id) => IsId(id) ? Path.Combine(UploadsDirectory, id) : null;

    /// <summary>
    /// The queued or running job that reads upload <paramref name="id"/> by path, or <c>null</c>. Removing the data under such a
    /// job would fail it minutes in, so neither <c>DELETE /datasets/{id}</c> nor the retention sweep does.
    /// </summary>
    public TrainingJob? JobReading(string id, TrainingJobStore jobs)
    {
        var folder = FolderOf(id);
        return folder is null ? null : jobs.GetAllJobs().FirstOrDefault(job =>
            job.Status is JobStatus.Queued or JobStatus.Running
            && new[] { job.DataFile }.Concat(job.AdditionalDataFiles).Any(file => ProjectPaths.IsWithin(Path.GetFullPath(file), folder)));
    }

    /// <summary>When upload <paramref name="id"/> was made — the second its id names, UTC — or <c>null</c> for an id that is not one.</summary>
    public static DateTime? CreatedAtOf(string id) => IsId(id)
        && DateTime.TryParseExact(id.AsSpan(3, 15), TimestampKeys.StampFormat, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var at) ? at : null;

    /// <summary>
    /// Removes the uploads older than <paramref name="days"/> days at <paramref name="now"/>, except one a queued or running
    /// job reads. The age is the id's, not a file's modification time: copying or touching the files does not renew an upload.
    /// </summary>
    public DatasetPruneReport Prune(int days, DateTimeOffset now, TrainingJobStore jobs)
    {
        var cutoff = now.UtcDateTime.AddDays(-days);
        int removed = 0, kept = 0;
        long bytes = 0;
        foreach (var dataset in List())
        {
            if (CreatedAtOf(dataset.Id) is not { } madeAt || madeAt >= cutoff)
                continue;
            if (JobReading(dataset.Id, jobs) is not null)
            {
                kept++;
                continue;
            }
            Directory.Delete(Path.Combine(UploadsDirectory, dataset.Id), recursive: true);
            removed++;
            bytes += dataset.Bytes;
        }
        return new DatasetPruneReport(removed, kept, bytes, cutoff);
    }

    /// <summary>Removes one upload; <c>false</c> when there is none by that id.</summary>
    public bool Delete(string id)
    {
        if (Find(id) is null)
            return false;
        Directory.Delete(Path.Combine(UploadsDirectory, id), recursive: true);
        return true;
    }

    private StoredDataset? Describe(string folder)
    {
        var id = Path.GetFileName(folder);
        var children = Directory.GetFileSystemEntries(folder);
        if (children.Length == 0)
            return null;
        // A table upload is one file; a zip's content is `content/`, or the one folder the zip held.
        string path;
        string kind;
        if (children.Length == 1 && File.Exists(children[0]))
        {
            path = children[0];
            kind = "table";
        }
        else
        {
            var content = Path.Combine(folder, "content");
            var inner = Directory.Exists(content) ? Directory.GetFileSystemEntries(content) : [];
            path = inner.Length == 1 && Directory.Exists(inner[0]) ? inner[0] : (Directory.Exists(content) ? content : folder);
            kind = "directory";
        }
        var (files, bytes) = Count(path);
        var relative = Path.GetRelativePath(projectRoot, path).Replace(Path.DirectorySeparatorChar, '/');
        return new StoredDataset(id, kind, relative, files, bytes, CreatedAtOf(id) ?? Directory.GetCreationTimeUtc(folder));
    }

    private static (int Files, long Bytes) Count(string path)
    {
        if (File.Exists(path))
            return (1, new FileInfo(path).Length);
        var files = Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories).ToList();
        return (files.Count, files.Sum(f => new FileInfo(f).Length));
    }

    /// <summary>The file name an upload names: one plain name that is the same on every file system the project may sit on.</summary>
    internal static string CheckFileName(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            throw new DatasetRefusedException("Name the file: ?fileName=<name>.<csv|tsv|xlsx|parquet|jsonl|zip …>.");
        var name = fileName.Trim();
        if (name.Length > 200 || name.StartsWith('.') || name != Path.GetFileName(name)
            || name.Any(c => char.IsControl(c) || "<>:\"/\\|?*".Contains(c)))
            throw new DatasetRefusedException(
                $"'{fileName}' is not a file name the server will store: a plain name of at most 200 characters, with no path, control character or any of < > : \" / \\ | ? *, not beginning with a dot.");
        return name;
    }

    /// <summary>The path of a zip entry made safe to join under the unpacked folder, or the reason it is refused.</summary>
    internal static string? EntryRefusal(string entryName)
    {
        if (entryName.Length == 0 || entryName.Contains('\\') || entryName.StartsWith('/')
            || entryName.Contains(':') || entryName.Any(char.IsControl))
            return $"The zip's entry '{entryName}' has a name the server will not unpack (an absolute path, a drive, a backslash or a control character).";
        if (entryName.TrimEnd('/').Split('/').Any(segment => segment is "" or "." or ".."))
            return $"The zip's entry '{entryName}' leaves the dataset's folder or is not a plain path.";
        return null;
    }

    private static async Task UnpackAsync(
        string archivePath, string destination, DatasetLimits limits, long expandedLimit, string expandedSetting, CancellationToken ct)
    {
        ZipArchive zip;
        try
        {
            zip = ZipFile.OpenRead(archivePath);
        }
        catch (InvalidDataException)
        {
            throw new DatasetRefusedException("The file is not a zip archive.");
        }
        using (zip)
        {
            if (zip.Entries.Count > limits.MaxEntries)
                throw new DatasetRefusedException(
                    $"The zip holds {zip.Entries.Count.ToString("N0", CultureInfo.InvariantCulture)} entries; at most {limits.MaxEntries.ToString("N0", CultureInfo.InvariantCulture)} are taken (Serve:Datasets:MaxEntries).",
                    StatusCodes.Status413PayloadTooLarge);

            // Every entry is judged before the first is written: a zip with one bad entry unpacks nothing.
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var entry in zip.Entries)
            {
                if (EntryRefusal(entry.FullName) is { } refusal)
                    throw new DatasetRefusedException(refusal);
                var isDirectory = entry.FullName.EndsWith('/');
                // The Unix file type sits in the high bits of the external attributes: 0xA000 is a link, 0x8000 a regular file.
                var unixType = (entry.ExternalAttributes >> 16) & 0xF000;
                if (unixType is not (0 or 0x8000 or 0x4000))
                    throw new DatasetRefusedException($"The zip's entry '{entry.FullName}' is a link or a special file; only regular files are taken.");
                if (!seen.Add(entry.FullName.TrimEnd('/')))
                    throw new DatasetRefusedException($"The zip holds '{entry.FullName.TrimEnd('/')}' twice (names are compared without regard to case).");
                if (isDirectory)
                    continue;
                var extension = Path.GetExtension(entry.Name);
                if (!ZipFileExtensions.Contains(extension))
                    throw new DatasetRefusedException(
                        $"The zip's entry '{entry.FullName}' is not a file a dataset holds ({string.Join(" ", ZipFileExtensions.Order())}).");
                if (entry.Length > expandedLimit)
                    throw new DatasetRefusedException($"The zip's entry '{entry.FullName}' alone unpacks beyond the limit ({expandedSetting}).", StatusCodes.Status413PayloadTooLarge);
            }

            long written = 0;
            foreach (var entry in zip.Entries)
            {
                ct.ThrowIfCancellationRequested();
                var target = Path.GetFullPath(Path.Combine(destination, entry.FullName));
                if (!ProjectPaths.IsWithin(target, destination))
                    throw new DatasetRefusedException($"The zip's entry '{entry.FullName}' leaves the dataset's folder.");
                if (entry.FullName.EndsWith('/'))
                {
                    Directory.CreateDirectory(target);
                    continue;
                }
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                await using var input = entry.Open();
                await using var output = File.Create(target);
                // The bytes that come out are counted, not the size the entry declares.
                written += await CopyLimitedAsync(input, output, expandedLimit - written, "The unpacked zip", expandedSetting, ct);
            }
        }
    }

    /// <summary>Copies up to <paramref name="limit"/> bytes; more is a 413 and the caller removes what was written.</summary>
    private static async Task<long> CopyLimitedAsync(Stream source, Stream destination, long limit, string what, string setting, CancellationToken ct)
    {
        var buffer = new byte[81920];
        long total = 0;
        int read;
        while ((read = await source.ReadAsync(buffer, ct)) > 0)
        {
            total += read;
            if (total > limit)
                throw new DatasetRefusedException(
                    $"{what} is larger than the server takes ({limit.ToString("N0", CultureInfo.InvariantCulture)} bytes; {setting}).",
                    StatusCodes.Status413PayloadTooLarge);
            await destination.WriteAsync(buffer.AsMemory(0, read), ct);
        }
        return total;
    }

    private static void TryDelete(string folder)
    {
        try
        {
            if (Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }
        catch (IOException)
        {
            // What the failure left stays under the hidden `.incoming-` name, which no listing shows.
        }
    }
}
