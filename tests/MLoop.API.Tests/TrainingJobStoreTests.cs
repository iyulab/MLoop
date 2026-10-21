using System.Net;
using System.Text.Json;
using FluentAssertions;
using MLoop.API;

namespace MLoop.API.Tests;

/// <summary>
/// The training jobs outlive the server process: a client polling <c>GET /jobs/{id}</c> across a
/// restart still finds its job, and the job's state says what became of it.
/// </summary>
public class TrainingJobStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-jobs-{Guid.NewGuid():N}");

    public TrainingJobStoreTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private TrainingJobStore Open() => new(new ProjectRootPath(_root));

    private static TrainingJobRequest Request(string dataFile = "datasets/train.csv") =>
        new(dataFile, "label", "binary-classification", Name: "churn", TimeLimitSeconds: 30, Weight: "w", Balance: "auto");

    [Fact]
    public void A_job_and_its_result_survive_a_restart()
    {
        var first = Open();
        var id = first.CreateJob(Request());
        first.UpdateStatus(id, JobStatus.Running, "Training started");
        first.SetResult(id, "exp-003", new Dictionary<string, double> { ["auc"] = 0.91 }, "FastTree");
        first.UpdateStatus(id, JobStatus.Completed, "Training completed");

        var job = Open().GetJob(id);

        job.Should().NotBeNull();
        job!.Status.Should().Be(JobStatus.Completed);
        job.StatusMessage.Should().Be("Training completed");
        job.CompletedAt.Should().NotBeNull();
        job.ExperimentId.Should().Be("exp-003");
        job.Metrics.Should().ContainKey("auc").WhoseValue.Should().Be(0.91);
        job.BestTrainer.Should().Be("FastTree");
        job.ModelName.Should().Be("churn");
        job.TimeLimitSeconds.Should().Be(30);
        job.Weight.Should().Be("w");
        job.Balance.Should().Be("auto");
    }

    [Fact]
    public void A_job_the_previous_process_left_running_reads_as_interrupted()
    {
        var first = Open();
        var id = first.CreateJob(Request());
        first.UpdateStatus(id, JobStatus.Running, "Training started");

        var job = Open().GetJob(id)!;

        job.Status.Should().Be(JobStatus.Failed);
        job.StatusMessage.Should().Be(TrainingJobStore.InterruptedMessage);
        job.CompletedAt.Should().NotBeNull();
        // And the record says so too, not only this process's memory.
        Open().GetJob(id)!.Status.Should().Be(JobStatus.Failed);
    }

    [Fact]
    public void Queued_jobs_stay_queued_in_the_order_they_were_accepted()
    {
        var first = Open();
        var older = first.CreateJob(Request());
        Thread.Sleep(20);
        var newer = first.CreateJob(Request());
        first.UpdateStatus(newer, JobStatus.Queued);

        Open().GetQueuedJobs().Select(j => j.JobId).Should().Equal(older, newer);
    }

    [Fact]
    public void An_unreadable_record_is_skipped_rather_than_stopping_the_server()
    {
        var id = Open().CreateJob(Request());
        var directory = Path.Combine(_root, ".mloop", TrainingJobStore.JobsDirectoryName);
        File.WriteAllText(Path.Combine(directory, "job-broken.json"), "{ not json");
        File.WriteAllText(Path.Combine(directory, "job-other.json"),
            File.ReadAllText(Path.Combine(directory, id + ".json"))); // names a different job

        var store = Open();

        store.GetAllJobs().Select(j => j.JobId).Should().Equal(id);
    }

    [Fact]
    public void The_record_is_readable_json_with_named_states()
    {
        var id = Open().CreateJob(Request());

        using var doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(_root, ".mloop", TrainingJobStore.JobsDirectoryName, id + ".json")));

        doc.RootElement.GetProperty("jobId").GetString().Should().Be(id);
        doc.RootElement.GetProperty("status").GetString().Should().Be("Queued");
        doc.RootElement.TryGetProperty("syncRoot", out _).Should().BeFalse();
    }
}

/// <summary>A server that starts on a project holding a queued job trains it.</summary>
public class QueuedJobOnStartupTests : IDisposable
{
    private readonly TestWebApplicationFactory _factory = new();

    public void Dispose() => _factory.Dispose();

    [Fact]
    public async Task A_job_queued_before_the_restart_is_picked_up()
    {
        // A job the previous process accepted and never started. Its data is gone, so training
        // fails fast — what matters is that the runner reached it.
        var seeded = new TrainingJobStore(new ProjectRootPath(_factory.TestProjectRoot));
        var id = seeded.CreateJob(new TrainingJobRequest(
            Path.Combine(_factory.TestProjectRoot, "datasets", "missing.csv"), "label", "binary-classification"));

        var client = _factory.CreateClient();
        JsonElement job = default;
        for (var i = 0; i < 100; i++)
        {
            var response = await client.GetAsync($"/jobs/{id}");
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            job = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
            if (job.GetProperty("status").GetString() == "Failed") break;
            await Task.Delay(100);
        }

        job.GetProperty("status").GetString().Should().Be("Failed");
        job.GetProperty("statusMessage").GetString().Should().NotBe(TrainingJobStore.InterruptedMessage);
    }
}
