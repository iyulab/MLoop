using MLoop.Core.Diagnostics;

namespace MLoop.Core.Tests.Diagnostics;

/// <summary>
/// Reported by a consumer: <c>mloop train --data data.csv</c> finished by warning "Standard data
/// file(s) not used: data.csv" — the very file it had just trained on. The scan was right (the
/// unused file was a second copy under <c>datasets/</c>); the display collapsed two distinct files
/// into one name.
/// </summary>
public class UnusedDataNameCollisionTests : IDisposable
{
    private readonly string _root;

    public UnusedDataNameCollisionTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "mloop-unused-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_root, "datasets"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private string Write(string relativePath, string content = "a,b\n1,2\n")
    {
        var path = Path.Combine(_root, relativePath);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void A_name_shared_with_a_used_file_is_shown_as_a_path()
    {
        var used = Write("data.csv");
        Write(Path.Combine("datasets", "data.csv"));

        var result = new UnusedDataScanner().Scan(Path.Combine(_root, "datasets"), [used]);

        var unused = Assert.Single(result.UnusedFiles);
        Assert.Equal("datasets/data.csv", unused.DisplayName);
    }

    [Fact]
    public void The_warning_says_a_same_named_file_was_used_so_it_does_not_read_as_a_contradiction()
    {
        var used = Write("data.csv");
        Write(Path.Combine("datasets", "data.csv"));

        var result = new UnusedDataScanner().Scan(Path.Combine(_root, "datasets"), [used]);

        Assert.Contains(result.Warnings, w => w.Contains("Standard data file(s) not used: datasets/data.csv"));
        Assert.Contains(result.Warnings, w => w.Contains("A different file with the same name was used"));
    }

    [Fact]
    public void An_unambiguous_name_is_still_shown_bare()
    {
        // Paths everywhere would be noise: the collision is what justifies them.
        var used = Write(Path.Combine("datasets", "train.csv"));
        Write(Path.Combine("datasets", "leftover.csv"));

        var result = new UnusedDataScanner().Scan(Path.Combine(_root, "datasets"), [used]);

        var unused = Assert.Single(result.UnusedFiles);
        Assert.Equal("leftover.csv", unused.DisplayName);
        Assert.DoesNotContain(result.Warnings, w => w.Contains("A different file with the same name was used"));
    }

    [Fact]
    public void The_recursive_scan_names_files_the_same_way()
    {
        var used = Write("data.csv");
        Write(Path.Combine("datasets", "data.csv"));

        var result = new UnusedDataScanner().ScanRecursive(Path.Combine(_root, "datasets"), [used]);

        Assert.Equal("datasets/data.csv", Assert.Single(result.UnusedFiles).DisplayName);
    }
}
