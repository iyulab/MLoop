using MLoop.Search;

namespace MLoop.Search.Tests;

public class ColumnRolesTests
{
    private static ColumnRoles Roles(params (string Name, ColumnRole Role)[] columns) =>
        new("Label", columns.Select(c => KeyValuePair.Create(c.Name, c.Role)));

    [Fact]
    public void EachRolesColumns_KeepTheirOrder()
    {
        var roles = Roles(("a", ColumnRole.Numeric), ("t", ColumnRole.Text), ("b", ColumnRole.Numeric));

        Assert.Equal(["a", "b"], roles.Of(ColumnRole.Numeric));
        Assert.Equal(["t"], roles.Of(ColumnRole.Text));
        Assert.Empty(roles.Of(ColumnRole.Ignored));
    }

    [Fact]
    public void ARoleChange_MovesTheColumnToTheEnd_AndLeavesTheRestInPlace()
    {
        // The order of a role's columns is the order they reach the feature vector: a text column
        // that becomes numeric joins after the columns that were numeric all along.
        var roles = Roles(("t", ColumnRole.Text), ("a", ColumnRole.Numeric), ("b", ColumnRole.Numeric));

        var changed = roles.With("t", ColumnRole.Numeric).With("t__keys", ColumnRole.Ignored);

        Assert.Equal(["a", "b", "t"], changed.Of(ColumnRole.Numeric));
        Assert.Equal(["t__keys"], changed.Of(ColumnRole.Ignored));
        Assert.Equal(ColumnRole.Text, roles.RoleOf("t")); // the original is unchanged
    }

    [Fact]
    public void AColumnHasOneRole()
    {
        var roles = Roles(("a", ColumnRole.Text), ("a", ColumnRole.Numeric));

        Assert.Single(roles.Columns);
        Assert.Equal(ColumnRole.Numeric, roles.RoleOf("a"));
        Assert.Null(roles.RoleOf("absent"));
    }

    [Fact]
    public void ALabelAndColumnNames_AreRequired()
    {
        Assert.Throws<ArgumentException>(() => new ColumnRoles(" ", []));
        Assert.Throws<ArgumentException>(() => Roles(("", ColumnRole.Numeric)));
    }
}
