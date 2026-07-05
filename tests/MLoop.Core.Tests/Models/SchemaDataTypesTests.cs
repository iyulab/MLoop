using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// Pins the single-source DataType vocabulary (upstream-008): producers
/// (CLI TrainingEngine, Core CaptureInputSchema) and consumers
/// (PredictionService, CsvDataLoader, CategoricalMapper) must share one table.
/// </summary>
public class SchemaDataTypesTests
{
    [Theory]
    [InlineData(typeof(float))]
    [InlineData(typeof(double))]
    [InlineData(typeof(int))]
    [InlineData(typeof(long))]
    public void FromRawType_maps_numeric_clr_types_to_Numeric(Type rawType)
    {
        Assert.Equal(SchemaDataTypes.Numeric, SchemaDataTypes.FromRawType(rawType));
    }

    [Fact]
    public void FromRawType_maps_bool_to_Boolean()
    {
        Assert.Equal(SchemaDataTypes.Boolean, SchemaDataTypes.FromRawType(typeof(bool)));
    }

    [Theory]
    [InlineData(typeof(string))]
    [InlineData(typeof(ReadOnlyMemory<char>))] // TextDataViewType.RawType
    public void FromRawType_maps_text_clr_types_to_Text(Type rawType)
    {
        Assert.Equal(SchemaDataTypes.Text, SchemaDataTypes.FromRawType(rawType));
    }

    [Fact]
    public void FromRawType_maps_unknown_types_to_Text()
    {
        Assert.Equal(SchemaDataTypes.Text, SchemaDataTypes.FromRawType(typeof(DateTime)));
    }

    [Theory]
    [InlineData("Numeric", DataKind.Single)]
    [InlineData("Categorical", DataKind.String)]
    [InlineData("Text", DataKind.String)]
    [InlineData("Boolean", DataKind.Boolean)]
    public void ToDataKind_maps_semantic_vocabulary(string dataType, DataKind expected)
    {
        Assert.Equal(expected, SchemaDataTypes.ToDataKind(dataType, fallback: DataKind.String));
    }

    [Theory]
    [InlineData("Single", DataKind.Single)]
    [InlineData("Double", DataKind.Double)]
    [InlineData("Int32", DataKind.Int32)]
    [InlineData("Int64", DataKind.Int64)]
    [InlineData("String", DataKind.String)] // BUG-42 tolerance
    public void ToDataKind_tolerates_legacy_raw_dotnet_names(string dataType, DataKind expected)
    {
        // Schemas captured before the vocabulary was unified persisted raw .NET
        // type names; they must still resolve instead of falling through.
        Assert.Equal(expected, SchemaDataTypes.ToDataKind(dataType, fallback: DataKind.String));
    }

    [Fact]
    public void ToDataKind_returns_fallback_for_unknown_names()
    {
        Assert.Equal(DataKind.Double, SchemaDataTypes.ToDataKind("Mystery", fallback: DataKind.Double));
    }
}
