using System.Diagnostics;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// "Each command runs independently and exits" means two <c>mloop train</c> processes can ask for an experiment id at the same
/// moment. The id was read from <c>experiment-index.json</c>, incremented and written back with nothing held between the three,
/// so both were given <c>exp-001</c>: one then failed on the other's <c>model.zip</c>, or — had they not overlapped there —
/// overwritten the other's records without a word. Threads of one process were safe (a lock in the process); processes were not,
/// so the test starts real ones.
/// </summary>
public class ExperimentIdAcrossProcessesTests
{
    private const string ProjectVariable = "MLOOP_TEST_ID_PROJECT";
    private const string OutVariable = "MLOOP_TEST_ID_OUT";
    private const string GoVariable = "MLOOP_TEST_ID_GO";
    private const int IdsPerProcess = 40;

    [Fact]
    public async Task ProcessesAskingForIdsAtTheSameTime_AreNeverGivenTheSameOne()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable(ProjectVariable) is not null, "this is the helper process");

        var root = Path.Combine(Path.GetTempPath(), "mloop-ids-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, ".mloop"));
        var go = Path.Combine(root, "go");
        try
        {
            var workers = Enumerable.Range(0, 3).Select(i => Start(root, Path.Combine(root, $"ids-{i}.txt"), go)).ToList();

            // Both are waiting at the start line; releasing them together is what makes them overlap.
            await Task.Delay(1500);
            await File.WriteAllTextAsync(go, "go");
            foreach (var worker in workers)
            {
                Assert.True(worker.WaitForExit(120_000), "a helper process did not finish");
                Assert.True(worker.ExitCode == 0, $"a helper process failed (exit {worker.ExitCode}): {await worker.StandardOutput.ReadToEndAsync()}");
            }

            var ids = Enumerable.Range(0, 3).SelectMany(i => File.ReadAllLines(Path.Combine(root, $"ids-{i}.txt"))).ToList();

            Assert.Equal(3 * IdsPerProcess, ids.Count);
            Assert.Equal(ids.Count, ids.Distinct().Count());
            // No id is skipped either: the index and the processes agree on what has been given out.
            Assert.Equal(
                Enumerable.Range(1, ids.Count).Select(n => $"exp-{n:D3}"),
                ids.OrderBy(id => id, StringComparer.Ordinal));
        }
        finally
        {
            try { Directory.Delete(root, recursive: true); } catch (IOException) { }
        }
    }

    /// <summary>The helper the test above starts: asks for ids in a loop and writes them down. Not a test of its own.</summary>
    [Fact]
    public async Task HelperProcess_AsksForIds()
    {
        var root = Environment.GetEnvironmentVariable(ProjectVariable);
        Assert.SkipWhen(root is null, "started by ProcessesAskingForIdsAtTheSameTime_AreNeverGivenTheSameOne");

        var go = Environment.GetEnvironmentVariable(GoVariable)!;
        for (var waited = 0; !File.Exists(go) && waited < 600; waited++)
            await Task.Delay(50);

        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem), root!);
        var ids = new List<string>();
        for (var i = 0; i < IdsPerProcess; i++)
            ids.Add(await store.GenerateIdAsync("default", CancellationToken.None));
        await File.WriteAllLinesAsync(Environment.GetEnvironmentVariable(OutVariable)!, ids);
    }

    private static Process Start(string root, string output, string go)
    {
        var entry = Environment.GetCommandLineArgs()[0];
        var self = Environment.ProcessPath!;
        var host = Path.GetFileNameWithoutExtension(self).Equals("dotnet", StringComparison.OrdinalIgnoreCase);
        var info = new ProcessStartInfo(self)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        if (host)
            info.ArgumentList.Add(entry);
        info.ArgumentList.Add("-method");
        info.ArgumentList.Add($"{typeof(ExperimentIdAcrossProcessesTests).FullName}.{nameof(HelperProcess_AsksForIds)}");
        info.Environment[ProjectVariable] = root;
        info.Environment[OutVariable] = output;
        info.Environment[GoVariable] = go;
        return Process.Start(info)!;
    }
}
