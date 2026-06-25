using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class FeaturesCommandTests
{
    [Fact]
    public void ReadHeaderColumns_ReturnsHeader_AndDoesNotMutateFile()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-feat-" + Guid.NewGuid());
        Directory.CreateDirectory(dir);
        try
        {
            var csv = Path.Combine(dir, "t.csv");
            File.WriteAllText(csv, "a,b,y\n1,2,3\n4,5,6\n");
            var before = File.GetLastWriteTimeUtc(csv);

            var cols = FeaturesCommand.ReadHeaderColumns(csv);

            Assert.Equal(new[] { "a", "b", "y" }, cols);
            Assert.Equal(before, File.GetLastWriteTimeUtc(csv)); // 데이터 불변(P2)
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
