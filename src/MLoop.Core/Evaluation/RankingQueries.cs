using Microsoft.ML;

namespace MLoop.Core.Evaluation;

/// <summary>
/// The queries a ranking is scored over: the scored file's own. A model maps its group column to a key when it is fitted,
/// and that map knows the queries training saw; a query it never saw maps to the missing key. A held-out query is exactly
/// that, so scoring by the model's key put every held-out query into one group - one list of all the rows, whose NDCG and
/// DCG are those of a single list, not of queries. Training's own test partition and a file given to <c>evaluate</c>
/// are both made of held-out queries, so both map the group column afresh here.
/// </summary>
public static class RankingQueries
{
    /// <summary>The key column <see cref="Of"/> adds.</summary>
    public const string Column = "EvaluatedQuery";

    /// <summary>
    /// The scored rows with a key column that is one value per query of these rows, and its name. Where the rows have no
    /// <paramref name="groupColumn"/>, the model's own <paramref name="fallback"/> key (or the group column itself).
    /// </summary>
    public static (IDataView Scored, string KeyColumn) Of(MLContext ml, IDataView scored, string? groupColumn, string fallback = "GroupId")
    {
        if (!string.IsNullOrEmpty(groupColumn) && scored.Schema.GetColumnOrNull(groupColumn) is not null)
            return (ml.Transforms.Conversion.MapValueToKey(Column, groupColumn).Fit(scored).Transform(scored), Column);

        if (scored.Schema.GetColumnOrNull(fallback) is null && !string.IsNullOrEmpty(groupColumn))
            return (scored, groupColumn);
        return (scored, fallback);
    }
}
