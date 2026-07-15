using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>Schema helpers shared by DL text handlers (moved out of AutoMLRunner for cross-assembly reuse).</summary>
public static class TextColumnFinder
{
    public static string? FindFirst(DataViewSchema schema, string labelColumn)
    {
        foreach (var col in schema)
        {
            if (col.IsHidden) continue;
            if (col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)) continue;
            if (col.Type is TextDataViewType) return col.Name;
        }
        return null;
    }

    public static List<string> Find(DataViewSchema schema, string labelColumn, int maxCount)
    {
        var result = new List<string>();
        foreach (var col in schema)
        {
            if (col.IsHidden) continue;
            if (col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)) continue;
            if (col.Type is TextDataViewType)
            {
                result.Add(col.Name);
                if (result.Count >= maxCount) break;
            }
        }
        return result;
    }
}
