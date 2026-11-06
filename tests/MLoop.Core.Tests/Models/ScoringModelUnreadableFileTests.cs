using Microsoft.ML;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// A model file that is there and is not a whole archive (cut short by a full disk or a killed copy, empty) was reported as
/// "Failed to open a zip archive / End of Central Directory record could not be found" - correct, and naming neither the
/// file nor the way back. It now says which file, and what restores a model.
/// </summary>
public class ScoringModelUnreadableFileTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-model-" + Guid.NewGuid().ToString("N"));

    public ScoringModelUnreadableFileTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
        GC.SuppressFinalize(this);
    }

    [Theory]
    [InlineData("")]
    [InlineData("this is not a zip archive")]
    [InlineData("PK\u0003\u0004 cut short")]
    public void AModelFileThatIsNotAWholeArchive_NamesTheFileAndTheWayBack(string content)
    {
        var path = Path.Combine(_dir, ExperimentLayout.ModelFileName);
        File.WriteAllText(path, content);

        var ex = Assert.Throws<UnreadableStateFileException>(() => ScoringModel.Load(new MLContext(seed: 1), path));

        Assert.Equal(path, ex.Path);
        Assert.Contains(path, ex.Message, StringComparison.Ordinal);
        Assert.Contains("mloop promote", ex.Message, StringComparison.Ordinal);
        Assert.Contains("mloop rollback", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingModelFile_IsStillAMissingFile_NotDamage()
    {
        var path = Path.Combine(_dir, "absent", ExperimentLayout.ModelFileName);

        Assert.False(Record.Exception(() => ScoringModel.Load(new MLContext(seed: 1), path)) is UnreadableStateFileException);
    }
}
