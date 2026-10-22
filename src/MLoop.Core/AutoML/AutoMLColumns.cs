using Microsoft.ML.AutoML;
using MLoop.Search;

namespace MLoop.Core.AutoML;

/// <summary>
/// Hands MLoop's column roles to ML.NET AutoML in the shape it takes. The only place a
/// <see cref="ColumnRoles"/> becomes an AutoML <see cref="ColumnInformation"/>; it goes when
/// Microsoft.ML.AutoML does.
/// </summary>
internal static class AutoMLColumns
{
    /// <summary>
    /// True when the roles name a text, categorical or ignored column. Without one, AutoML is given
    /// the label alone and infers the rest: explicit column information that lists only numeric
    /// columns made it build an empty feature vector and fail.
    /// </summary>
    public static bool NeedsExplicitRoles(ColumnRoles roles) =>
        roles.Of(ColumnRole.Text).Count > 0
        || roles.Of(ColumnRole.Categorical).Count > 0
        || roles.Of(ColumnRole.Ignored).Count > 0;

    /// <summary>The roles as AutoML column information, or null when AutoML should infer them.</summary>
    public static ColumnInformation? ToColumnInformation(ColumnRoles roles)
    {
        if (!NeedsExplicitRoles(roles))
            return null;

        var info = new ColumnInformation { LabelColumnName = roles.Label };
        foreach (var name in roles.Of(ColumnRole.Text))
            info.TextColumnNames.Add(name);
        foreach (var name in roles.Of(ColumnRole.Numeric))
            info.NumericColumnNames.Add(name);
        foreach (var name in roles.Of(ColumnRole.Categorical))
            info.CategoricalColumnNames.Add(name);
        foreach (var name in roles.Of(ColumnRole.Ignored))
            info.IgnoredColumnNames.Add(name);
        return info;
    }
}
