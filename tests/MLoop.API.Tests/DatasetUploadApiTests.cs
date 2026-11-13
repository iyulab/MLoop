using System.IO.Compression;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /datasets</c>: training data over HTTP. A table file or a zip of a directory dataset is kept under the project's
/// <c>datasets/uploads/&lt;id&gt;/</c> and its project-relative path is what <c>POST /train</c> takes — and everything the server
/// will not put on its disk is refused with nothing left behind: a path that leaves the folder, a link, a file no dataset
/// holds, more bytes than the limits allow (counted as they arrive — a zip's declared sizes are the sender's word).
/// </summary>
public class DatasetUploadApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public DatasetUploadApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private string Uploads => Path.Combine(_factory.TestProjectRoot, "datasets", "uploads");

    private static ByteArrayContent Bytes(byte[] body) => new(body);

    private static byte[] Csv() => Encoding.UTF8.GetBytes("x,y\n" + string.Join('\n', Enumerable.Range(0, 40).Select(i => $"{i},{i * 2}")) + "\n");

    private static readonly byte[] Png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>A zip with the named entries (name → bytes); <paramref name="configure"/> adjusts an entry after it is made.</summary>
    private static byte[] Zip(Dictionary<string, byte[]> entries, Action<ZipArchiveEntry>? configure = null)
    {
        using var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, content) in entries)
            {
                var entry = zip.CreateEntry(name, CompressionLevel.Optimal);
                using (var writer = entry.Open())
                    writer.Write(content);
                configure?.Invoke(entry);
            }
        }
        return stream.ToArray();
    }

    private async Task<HttpResponseMessage> UploadAsync(string fileName, byte[] body, string? task = null, HttpClient? client = null)
    {
        var url = $"/datasets?fileName={Uri.EscapeDataString(fileName)}" + (task is null ? "" : $"&task={task}");
        return await (client ?? _client).PostAsync(url, Bytes(body));
    }

    private static async Task<JsonElement> JsonOf(HttpResponseMessage response) =>
        JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.Clone();

    private IEnumerable<string> StoredIds() =>
        Directory.Exists(Uploads) ? Directory.GetDirectories(Uploads).Select(Path.GetFileName)!.Where(n => n!.StartsWith("ds-")).Select(n => n!) : [];

    private IEnumerable<string> Leftovers() =>
        Directory.Exists(Uploads) ? Directory.GetFileSystemEntries(Uploads).Select(Path.GetFileName).Where(n => n!.StartsWith(".incoming-")).Select(n => n!) : [];

    [Fact]
    public async Task ATableFile_IsKeptInTheProject_AndItsPathTrains()
    {
        var response = await UploadAsync("sales.csv", Csv());

        response.StatusCode.Should().Be(HttpStatusCode.Created);
        var body = await JsonOf(response);
        var id = body.GetProperty("id").GetString()!;
        body.GetProperty("kind").GetString().Should().Be("table");
        var dataFile = body.GetProperty("dataFile").GetString()!;
        dataFile.Should().Be($"datasets/uploads/{id}/sales.csv", "the path is the project's, as POST /train and POST /samples speak of data");
        body.GetProperty("files").GetInt32().Should().Be(1);
        body.GetProperty("bytes").GetInt64().Should().Be(Csv().Length);
        response.Headers.Location!.ToString().Should().Be($"/datasets/{id}");
        File.ReadAllBytes(Path.Combine(_factory.TestProjectRoot, dataFile)).Should().Equal(Csv());

        // The path is the whole hand-off: the training request takes it as it is.
        var train = await _client.PostAsJsonAsync("/train", new { dataFile, labelColumn = "y", task = "regression", promote = false });
        train.StatusCode.Should().Be(HttpStatusCode.Accepted);
    }

    [Theory]
    [InlineData("application/x-www-form-urlencoded")] // what curl --data-binary sends unless told otherwise
    [InlineData("text/csv")]
    [InlineData("application/octet-stream")]
    [InlineData("application/zip")]
    public async Task ThePayload_IsTheFile_WhateverContentTypeItIsLabelled(string contentType)
    {
        var content = Bytes(Csv());
        content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(contentType);

        var response = await _client.PostAsync("/datasets?fileName=labelled.csv", content);

        response.StatusCode.Should().Be(HttpStatusCode.Created);
    }

    [Fact]
    public async Task AZipOfImageClassFolders_IsUnpackedAsTheDirectoryDataset_AndItsPathTrains()
    {
        // Zipped as a file manager does it: the folder itself is the zip's one root.
        var zip = Zip(new()
        {
            ["photos/ok/a.png"] = Png,
            ["photos/ok/b.png"] = Png,
            ["photos/ng/c.png"] = Png,
        });

        var response = await UploadAsync("photos.zip", zip, task: "image-classification");

        response.StatusCode.Should().Be(HttpStatusCode.Created);
        var body = await JsonOf(response);
        body.GetProperty("kind").GetString().Should().Be("directory");
        body.GetProperty("files").GetInt32().Should().Be(3);
        var dataFile = body.GetProperty("dataFile").GetString()!;
        dataFile.Should().EndWith("/photos", "the one folder the zip held is the dataset, as it would be on disk");
        Directory.Exists(Path.Combine(_factory.TestProjectRoot, dataFile, "ok")).Should().BeTrue();

        var train = await _client.PostAsJsonAsync("/train", new { dataFile, task = "image-classification", promote = false });
        train.StatusCode.Should().Be(HttpStatusCode.Accepted);
    }

    [Fact]
    public async Task AZipWithSeveralRoots_IsTheDatasetItself()
    {
        var response = await UploadAsync("flat.zip", Zip(new() { ["ok/a.png"] = Png, ["ng/b.png"] = Png }), task: "image-classification");

        response.StatusCode.Should().Be(HttpStatusCode.Created);
        var dataFile = (await JsonOf(response)).GetProperty("dataFile").GetString()!;
        dataFile.Should().EndWith("/content");
        Directory.Exists(Path.Combine(_factory.TestProjectRoot, dataFile, "ok")).Should().BeTrue();
    }

    [Theory]
    [InlineData("../evil.png", "leaves the dataset's folder")]
    [InlineData("a/../../evil.png", "leaves the dataset's folder")]
    [InlineData("/abs/evil.png", "will not unpack")]
    [InlineData("C:/evil.png", "will not unpack")]
    [InlineData("a\\evil.png", "will not unpack")]
    public async Task AZipEntryThatLeavesItsFolder_RefusesTheWholeZip_AndLeavesNothing(string entryName, string reason)
    {
        var before = StoredIds().ToHashSet();
        var zip = Zip(new() { ["ok/good.png"] = Png, [entryName] = Png });

        var response = await UploadAsync("bad.zip", zip);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain(reason);
        StoredIds().ToHashSet().Should().BeEquivalentTo(before, "a zip with one bad entry unpacks nothing");
        Leftovers().Should().BeEmpty();
        File.Exists(Path.Combine(_factory.TestProjectRoot, "evil.png")).Should().BeFalse();
    }

    [Fact]
    public async Task AZipEntryThatIsALink_IsRefused()
    {
        // 0xA000 is the Unix file type of a symbolic link, kept in the high half of the entry's external attributes.
        var zip = Zip(new() { ["ok/a.png"] = Png, ["ok/link.png"] = Encoding.UTF8.GetBytes("/etc/passwd") },
            entry => { if (entry.Name == "link.png") entry.ExternalAttributes = (0xA000 | 0x1FF) << 16; });

        var response = await UploadAsync("link.zip", zip);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("is a link or a special file");
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public async Task AFileNoDatasetHolds_IsRefused()
    {
        var response = await UploadAsync("payload.zip", Zip(new() { ["ok/a.png"] = Png, ["ok/run.exe"] = [0x4D, 0x5A] }));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("run.exe");
    }

    [Theory]
    [InlineData("", "Name the file")]
    [InlineData("..", "not a file name")]
    [InlineData("sub/dir.csv", "not a file name")]
    [InlineData("a\\b.csv", "not a file name")]
    [InlineData(".hidden.csv", "not a file name")]
    [InlineData("data.exe", "neither a table file")]
    public async Task AFileNameTheServerWillNotStore_IsRefused(string name, string reason)
    {
        var response = await _client.PostAsync(string.IsNullOrEmpty(name) ? "/datasets" : $"/datasets?fileName={Uri.EscapeDataString(name)}", Bytes(Csv()));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain(reason);
    }

    [Fact]
    public async Task AnEmptyBody_IsRefused_AndLeavesNothing()
    {
        var response = await UploadAsync("empty.csv", []);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("body is empty");
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public async Task ANamedFileThatIsNotAZip_IsRefused()
    {
        var response = await UploadAsync("notes.zip", Csv());

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("not a zip archive");
        Leftovers().Should().BeEmpty();
    }

    [Theory]
    [InlineData("photos.zip", "regression", "trains on a table")]
    [InlineData("table.csv", "image-classification", "trains on a directory dataset")]
    public async Task AnUploadForATaskItDoesNotFit_IsRefusedAtTheDoor(string file, string task, string reason)
    {
        var body = file.EndsWith(".zip") ? Zip(new() { ["ok/a.png"] = Png, ["ng/b.png"] = Png }) : Csv();

        var response = await UploadAsync(file, body, task);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain(reason);
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public async Task ADirectoryOfTheWrongLayoutForTheTask_IsRefusedWithTheTrainingRequestsReason_AndLeavesNothing()
    {
        var before = StoredIds().ToHashSet();

        // Images and nothing else: no class subfolders.
        var response = await UploadAsync("flat.zip", Zip(new() { ["a.png"] = Png, ["b.png"] = Png }), task: "image-classification");

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("No class subfolders found");
        StoredIds().ToHashSet().Should().BeEquivalentTo(before);
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public async Task AnUploadBeyondTheLimit_IsTooLarge_AndLeavesNothing()
    {
        using var small = _factory.WithWebHostBuilder(b => b.UseSetting("Serve:Datasets:MaxUploadBytes", "1024"));
        var client = small.CreateClient();

        var response = await UploadAsync("big.csv", Encoding.UTF8.GetBytes("x\n" + new string('1', 4096)), client: client);

        response.StatusCode.Should().Be(HttpStatusCode.RequestEntityTooLarge);
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public async Task AZipThatUnpacksBeyondTheLimit_IsTooLarge_WhateverItDeclares()
    {
        // 1 MB of zeros compresses to a few hundred bytes: small to send, large on the disk.
        using var limited = _factory.WithWebHostBuilder(b => b.UseSetting("Serve:Datasets:MaxExpandedBytes", "65536"));
        var client = limited.CreateClient();
        var zip = Zip(new() { ["ok/a.csv"] = new byte[1 << 20] });
        zip.Length.Should().BeLessThan(10_000, "the bomb is small on the wire");

        var response = await UploadAsync("bomb.zip", zip, client: client);

        response.StatusCode.Should().Be(HttpStatusCode.RequestEntityTooLarge);
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public async Task AZipWithMoreEntriesThanTheLimit_IsTooLarge()
    {
        using var limited = _factory.WithWebHostBuilder(b => b.UseSetting("Serve:Datasets:MaxEntries", "3"));
        var client = limited.CreateClient();

        var response = await UploadAsync("many.zip", Zip(Enumerable.Range(0, 5).ToDictionary(i => $"ok/{i}.png", _ => Png)), client: client);

        response.StatusCode.Should().Be(HttpStatusCode.RequestEntityTooLarge);
    }

    /// <summary>What the stored uploads add up to now — the other tests of this class share the project.</summary>
    private async Task<long> StoredTotalAsync() =>
        (await JsonOf(await _client.GetAsync("/datasets"))).GetProperty("datasets").EnumerateArray().Sum(d => d.GetProperty("bytes").GetInt64());

    [Fact]
    public async Task UploadsBeyondTheTotalLimit_AreTooLarge_UntilARemovalMakesRoom()
    {
        var csv = Csv();
        var stored = await StoredTotalAsync();
        using var limited = _factory.WithWebHostBuilder(b => b.UseSetting("Serve:Datasets:MaxTotalBytes", (stored + 2 * csv.Length + 10).ToString()));
        var client = limited.CreateClient();
        var before = StoredIds().ToHashSet();

        var first = await UploadAsync("a.csv", csv, client: client);
        var second = await UploadAsync("b.csv", csv, client: client);
        var third = await UploadAsync("c.csv", csv, client: client);

        first.StatusCode.Should().Be(HttpStatusCode.Created);
        second.StatusCode.Should().Be(HttpStatusCode.Created);
        third.StatusCode.Should().Be(HttpStatusCode.RequestEntityTooLarge);
        var message = await third.Content.ReadAsStringAsync();
        message.Should().Contain("Serve:Datasets:MaxTotalBytes", "the refusal names the setting that bound it");
        message.Should().Contain("DELETE /datasets/{id}", "and what frees room");
        StoredIds().Except(before).Should().HaveCount(2, "the refused upload left nothing");
        Leftovers().Should().BeEmpty();

        var firstId = (await JsonOf(first)).GetProperty("id").GetString()!;
        (await client.DeleteAsync($"/datasets/{firstId}")).StatusCode.Should().Be(HttpStatusCode.NoContent);
        (await UploadAsync("d.csv", csv, client: client)).StatusCode.Should().Be(HttpStatusCode.Created);
    }

    [Fact]
    public async Task AZipThatUnpacksBeyondTheRoomLeft_IsTooLarge_NamingTheTotalLimit()
    {
        var stored = await StoredTotalAsync();
        using var limited = _factory.WithWebHostBuilder(b => b.UseSetting("Serve:Datasets:MaxTotalBytes", (stored + 65536).ToString()));
        var client = limited.CreateClient();
        var zip = Zip(new() { ["ok/a.csv"] = new byte[1 << 20] });

        var response = await UploadAsync("bomb.zip", zip, client: client);

        response.StatusCode.Should().Be(HttpStatusCode.RequestEntityTooLarge);
        (await response.Content.ReadAsStringAsync()).Should().Contain("Serve:Datasets:MaxTotalBytes");
        Leftovers().Should().BeEmpty();
    }

    [Fact]
    public void TheTotalLimit_DefaultsToACeilingAboveTheLimitOfOneUpload()
    {
        var limits = DatasetLimits.From(new Microsoft.Extensions.Configuration.ConfigurationBuilder().Build());

        limits.MaxTotalBytes.Should().Be(DatasetLimits.DefaultMaxTotalBytes);
        limits.MaxTotalBytes.Should().BeGreaterThan(limits.MaxExpandedBytes);
    }

    [Fact]
    public async Task TheStoredUploads_AreListedNewestFirst_AndRemovedById()
    {
        var first = (await JsonOf(await UploadAsync("one.csv", Csv()))).GetProperty("id").GetString()!;
        await Task.Delay(1100); // ids carry the second they were made in
        var second = (await JsonOf(await UploadAsync("two.csv", Csv()))).GetProperty("id").GetString()!;

        var list = await JsonOf(await _client.GetAsync("/datasets"));
        var ids = list.GetProperty("datasets").EnumerateArray().Select(d => d.GetProperty("id").GetString()).ToList();
        ids.Should().Contain([first, second]);
        ids.IndexOf(second).Should().BeLessThan(ids.IndexOf(first), "newest first");
        list.GetProperty("count").GetInt32().Should().Be(ids.Count);

        (await _client.DeleteAsync($"/datasets/{first}")).StatusCode.Should().Be(HttpStatusCode.NoContent);
        Directory.Exists(Path.Combine(Uploads, first)).Should().BeFalse();
        (await _client.DeleteAsync($"/datasets/{first}")).StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Theory]
    [InlineData("..")]
    [InlineData("not-an-id")]
    [InlineData("ds-20260101-000000-zzzzzzzz")]
    public async Task OnlyAnIdTheStoreMade_IsLetIntoAPath(string id)
    {
        var response = await _client.DeleteAsync($"/datasets/{id}");

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
        Directory.Exists(Path.Combine(_factory.TestProjectRoot, "datasets")).Should().BeTrue("a bad id removes nothing");
    }

    [Fact]
    public async Task AnUploadAJobIsQueuedOn_IsNotRemoved_UntilTheJobIsDone()
    {
        var uploaded = await JsonOf(await UploadAsync("busy.csv", Csv()));
        var id = uploaded.GetProperty("id").GetString()!;
        var dataFile = Path.Combine(_factory.TestProjectRoot, uploaded.GetProperty("dataFile").GetString()!);
        // A job made in the store and never handed to the runner stays queued: nothing races the assertion.
        var jobs = (TrainingJobStore)_factory.Services.GetService(typeof(TrainingJobStore))!;
        var jobId = jobs.CreateJob(new TrainingJobRequest(dataFile, "y", "regression"));

        var held = await _client.DeleteAsync($"/datasets/{id}");

        held.StatusCode.Should().Be(HttpStatusCode.Conflict);
        (await held.Content.ReadAsStringAsync()).Should().Contain(jobId);
        Directory.Exists(Path.Combine(Uploads, id)).Should().BeTrue();

        jobs.GetJob(jobId)!.Status = JobStatus.Failed;
        (await _client.DeleteAsync($"/datasets/{id}")).StatusCode.Should().Be(HttpStatusCode.NoContent);
    }
}
