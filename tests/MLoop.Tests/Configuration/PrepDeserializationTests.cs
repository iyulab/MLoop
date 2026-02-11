using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Preprocessing;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.Tests.Configuration;

/// <summary>
/// Tests that mloop.yaml prep section deserializes correctly via YamlDotNet.
/// </summary>
public class PrepDeserializationTests
{
    private readonly IDeserializer _deserializer;

    public PrepDeserializationTests()
    {
        _deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();
    }

    [Fact]
    public void Deserialize_PrepWithFillMissing_ParsesCorrectly()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: fill-missing
                    columns:
                      - pH
                      - Temp
                    method: mean
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);

        Assert.NotNull(config);
        Assert.Single(config.Models);
        var model = config.Models["default"];
        Assert.NotNull(model.Prep);
        Assert.Single(model.Prep);

        var step = model.Prep[0];
        Assert.Equal("fill-missing", step.Type);
        Assert.NotNull(step.Columns);
        Assert.Equal(2, step.Columns.Count);
        Assert.Contains("pH", step.Columns);
        Assert.Contains("Temp", step.Columns);
        Assert.Equal("mean", step.Method);
    }

    [Fact]
    public void Deserialize_PrepWithNormalize_ParsesCorrectly()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: normalize
                    columns:
                      - pH
                    method: z-score
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        var step = config.Models["default"].Prep![0];

        Assert.Equal("normalize", step.Type);
        Assert.Equal("z-score", step.Method);
    }

    [Fact]
    public void Deserialize_PrepWithExtractDate_ParsesCorrectly()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: extract-date
                    column: ParsedTime
                    features:
                      - hour
                      - day_of_week
                    remove_original: true
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        var step = config.Models["default"].Prep![0];

        Assert.Equal("extract-date", step.Type);
        Assert.Equal("ParsedTime", step.Column);
        Assert.NotNull(step.Features);
        Assert.Equal(2, step.Features.Count);
        Assert.Contains("hour", step.Features);
        Assert.Contains("day_of_week", step.Features);
        Assert.True(step.RemoveOriginal);
    }

    [Fact]
    public void Deserialize_PrepWithMultipleSteps_ParsesInOrder()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: drop-duplicates
                    key_columns:
                      - LoT
                      - pH
                  - type: remove-columns
                    columns:
                      - unused_col
                  - type: normalize
                    columns:
                      - pH
                      - Temp
                    method: min-max
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        var prep = config.Models["default"].Prep;

        Assert.NotNull(prep);
        Assert.Equal(3, prep.Count);
        Assert.Equal("drop-duplicates", prep[0].Type);
        Assert.Equal("remove-columns", prep[1].Type);
        Assert.Equal("normalize", prep[2].Type);

        // Verify key_columns deserialization
        Assert.NotNull(prep[0].KeyColumns);
        Assert.Equal(2, prep[0].KeyColumns!.Count);
    }

    [Fact]
    public void Deserialize_PrepWithRenameColumns_ParsesMappingCorrectly()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: rename-columns
                    mapping:
                      old_name: new_name
                      messy_col: clean_col
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        var step = config.Models["default"].Prep![0];

        Assert.Equal("rename-columns", step.Type);
        Assert.NotNull(step.Mapping);
        Assert.Equal(2, step.Mapping.Count);
        Assert.Equal("new_name", step.Mapping["old_name"]);
        Assert.Equal("clean_col", step.Mapping["messy_col"]);
    }

    [Fact]
    public void Deserialize_PrepWithFilterRows_ParsesCorrectly()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: filter-rows
                    column: score
                    operator: ">="
                    value: "80"
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        var step = config.Models["default"].Prep![0];

        Assert.Equal("filter-rows", step.Type);
        Assert.Equal("score", step.Column);
        Assert.Equal(">=", step.Operator);
        Assert.Equal("80", step.Value);
    }

    [Fact]
    public void Deserialize_NoPrep_PrepIsNull()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        Assert.Null(config.Models["default"].Prep);
    }

    [Fact]
    public void Deserialize_PrepWithParseDatetime_ParsesCorrectly()
    {
        var yaml = """
            project: test-project
            models:
              default:
                task: regression
                label: Temp
                prep:
                  - type: parse-datetime
                    column: date_str
                    format: "yyyy-MM-dd HH:mm:ss"
            """;

        var config = _deserializer.Deserialize<MLoopConfig>(yaml);
        var step = config.Models["default"].Prep![0];

        Assert.Equal("parse-datetime", step.Type);
        Assert.Equal("date_str", step.Column);
        Assert.Equal("yyyy-MM-dd HH:mm:ss", step.Format);
    }
}
