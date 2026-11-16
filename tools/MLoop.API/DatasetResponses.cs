namespace MLoop.API;

// The bodies of the dataset endpoints (DatasetEndpoints), as the experiment ones are in ExperimentResponses.cs.

/// <summary>
/// An uploaded dataset (<c>POST /datasets</c>, and a row of <c>GET /datasets</c>): <c>dataFile</c> is its path in the project —
/// what <c>POST /train</c> takes as <c>dataFile</c> — and <c>kind</c> is <c>table</c> (one file) or <c>directory</c> (a zip's content).
/// </summary>
public sealed record DatasetResponse(string Id, string Kind, string DataFile, int Files, long Bytes, DateTime CreatedAt)
{
    internal static DatasetResponse Of(StoredDataset dataset) =>
        new(dataset.Id, dataset.Kind, dataset.Path, dataset.Files, dataset.Bytes, dataset.CreatedAt);
}

/// <summary><c>GET /datasets</c>: the stored uploads, newest first.</summary>
public sealed record DatasetListResponse(int Count, IReadOnlyList<DatasetResponse> Datasets);
