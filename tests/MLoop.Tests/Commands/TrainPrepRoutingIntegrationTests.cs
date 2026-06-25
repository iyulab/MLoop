using Microsoft.ML;
using MLoop.CLI.Commands;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands;

[Collection("FileSystem")]
public class TrainPrepRoutingIntegrationTests
{
    [Fact]
    public async Task ApplyPrepAsync_excludes_statistical_from_csv_and_emits_prefeaturizer()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"trainprep_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tmp);
        var input = Path.Combine(tmp, "train.csv");
        await File.WriteAllTextAsync(input, "age,city,label\n10,seoul,yes\n20,busan,no\n30,seoul,yes\n");

        var prep = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }, // preFeaturizer
            new() { Type = "remove-columns", Columns = new() { "city" } }                // csv
        };

        var ctx = new MLContext(seed: 42);
        var (dataFile, preFeaturizer, warnings, preFeaturizerColumns) =
            await TrainCommand.ApplyPrepAsync(input, prep, ctx, "binary-classification");

        var outText = await File.ReadAllTextAsync(dataFile);
        Assert.DoesNotContain("city", outText);  // remove-columns 적용
        Assert.Contains("10", outText);           // normalize 미적용(원값 유지 → 누수 차단)
        Assert.NotNull(preFeaturizer);            // normalize → preFeaturizer
        Assert.Empty(warnings);                   // 미지원 변환 없음
        Assert.Equal(new[] { "age" }, preFeaturizerColumns); // preFeaturizer가 참조하는 컬럼 → preserve 대상
        Directory.Delete(tmp, true);
    }

    [Fact]
    public async Task ApplyPrepAsync_bakes_statistical_into_csv_with_warning_for_non_prefeaturizer_task()
    {
        // clustering Execute site ignores config.PreFeaturizer; normalize must be applied in CSV
        // (not silently dropped) and a leakage warning emitted.
        var tmp = Path.Combine(Path.GetTempPath(), $"trainprep_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tmp);
        var input = Path.Combine(tmp, "train.csv");
        await File.WriteAllTextAsync(input, "age,city\n10,seoul\n20,busan\n30,seoul\n");

        var prep = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } },
            new() { Type = "remove-columns", Columns = new() { "city" } }
        };

        var ctx = new MLContext(seed: 42);
        var (dataFile, preFeaturizer, warnings, preFeaturizerColumns) =
            await TrainCommand.ApplyPrepAsync(input, prep, ctx, "clustering");

        var outText = await File.ReadAllTextAsync(dataFile);
        Assert.DoesNotContain("city", outText);   // remove-columns 적용
        Assert.DoesNotContain("10", outText);      // normalize CSV에 구워짐(원값 변경 → 적용됨)
        Assert.Null(preFeaturizer);                // 미소비 태스크 → preFeaturizer 없음
        Assert.Empty(preFeaturizerColumns);
        Assert.Single(warnings);                   // 누수 경고
        Directory.Delete(tmp, true);
    }
}
