using FluentAssertions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.API;

namespace MLoop.API.Tests;

/// <summary>
/// <c>Serve:Datasets:RetentionDays</c>: uploads older than the period are removed, once a day, by the age their id names —
/// not by a file's modification time, which copying renews — and never an upload a queued or running job reads. Without the
/// setting nothing is removed: an upload is training data someone may still mean to use.
/// </summary>
public class DatasetRetentionTests : IClassFixture<TestWebApplicationFactory>
{
    private const string OldId = "ds-20200101-000000-aaaaaaaa";
    private const string OlderBusyId = "ds-20190101-000000-bbbbbbbb";

    private readonly TestWebApplicationFactory _factory;

    public DatasetRetentionTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _ = factory.Services; // started, so the server's own first pass has run before a test writes its files
    }

    private string Uploads => Path.Combine(_factory.TestProjectRoot, "datasets", "uploads");

    private string Seed(string id)
    {
        var folder = Path.Combine(Uploads, id);
        Directory.CreateDirectory(folder);
        var file = Path.Combine(folder, "train.csv");
        File.WriteAllText(file, "x,y\n1,2\n");
        // The files are new: a pass that read modification times would keep every one of them.
        File.SetLastWriteTimeUtc(file, DateTime.UtcNow);
        return file;
    }

    private DatasetRetentionService Service(string? days) => new(
        _factory.Services.GetRequiredService<ProjectRootPath>(),
        new ConfigurationBuilder().AddInMemoryCollection(days is null ? [] : [KeyValuePair.Create(DatasetRetentionService.DaysKey, (string?)days)]).Build(),
        _factory.Services.GetRequiredService<TrainingJobStore>(),
        NullLogger<DatasetRetentionService>.Instance);

    [Fact]
    public void WithoutASetting_NothingIsRemoved()
    {
        var file = Seed(OldId);

        Service(null).PruneOnce().Should().BeNull();

        File.Exists(file).Should().BeTrue();
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-3")]
    public void ANonPositivePeriod_IsNotAPeriod_AndRemovesNothing(string days)
    {
        var file = Seed(OldId);

        Service(days).PruneOnce().Should().BeNull();

        File.Exists(file).Should().BeTrue();
    }

    [Fact]
    public async Task TheOlderUploads_AreRemoved_ByTheAgeTheirIdNames_AndTheRecentOnesStay()
    {
        var old = Seed(OldId);
        var client = _factory.CreateClient();
        var recent = await client.PostAsync("/datasets?fileName=new.csv", new ByteArrayContent("x\n1\n"u8.ToArray()));
        recent.EnsureSuccessStatusCode();
        var recentId = System.Text.Json.JsonDocument.Parse(await recent.Content.ReadAsStringAsync()).RootElement.GetProperty("id").GetString()!;

        var report = Service("30").PruneOnce();

        report.Should().NotBeNull();
        report!.Removed.Should().BeGreaterThanOrEqualTo(1);
        File.Exists(old).Should().BeFalse("the id says 2020");
        Directory.Exists(Path.Combine(Uploads, recentId)).Should().BeTrue("made now");
    }

    [Fact]
    public void AnUploadAJobReads_IsKept_UntilTheJobIsDone()
    {
        var file = Seed(OlderBusyId);
        var jobs = _factory.Services.GetRequiredService<TrainingJobStore>();
        var jobId = jobs.CreateJob(new TrainingJobRequest(file, "y", "regression"));

        var first = Service("30").PruneOnce();

        first!.KeptForJobs.Should().BeGreaterThanOrEqualTo(1);
        File.Exists(file).Should().BeTrue("a queued job reads it");

        jobs.GetJob(jobId)!.Status = JobStatus.Completed;
        Service("30").PruneOnce();

        File.Exists(file).Should().BeFalse("the job is done");
    }

    [Fact]
    public void AnIdThatIsNotOurs_IsNeverAgedOrRemoved()
    {
        var stranger = Path.Combine(Uploads, "keep-me");
        Directory.CreateDirectory(stranger);
        File.WriteAllText(Path.Combine(stranger, "a.csv"), "x\n");

        Service("1").PruneOnce();

        Directory.Exists(stranger).Should().BeTrue("only ids the store made are its to remove");
        DatasetStore.CreatedAtOf("keep-me").Should().BeNull();
        DatasetStore.CreatedAtOf(OldId).Should().Be(new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc));
    }
}
