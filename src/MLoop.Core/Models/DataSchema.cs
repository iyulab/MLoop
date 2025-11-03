using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// Represents the schema of a dataset
/// </summary>
public class DataSchema
{
    public required IReadOnlyList<ColumnInfo> Columns { get; init; }
    public required int RowCount { get; init; }
}

/// <summary>
/// Information about a single column
/// </summary>
public class ColumnInfo
{
    public required string Name { get; init; }
    public required string Type { get; init; }
    public required bool IsLabel { get; init; }
}

/// <summary>
/// Helper class to convert ML.NET types to friendly names
/// </summary>
public static class DataTypeHelper
{
    public static string GetFriendlyTypeName(DataViewType type)
    {
        if (type == NumberDataViewType.Single)
            return "float";
        if (type == NumberDataViewType.Double)
            return "double";
        if (type == NumberDataViewType.Int32)
            return "int";
        if (type == NumberDataViewType.Int64)
            return "long";
        if (type == NumberDataViewType.UInt32)
            return "uint";
        if (type == TextDataViewType.Instance)
            return "string";
        if (type == BooleanDataViewType.Instance)
            return "bool";

        return type.ToString() ?? "unknown";
    }
}
