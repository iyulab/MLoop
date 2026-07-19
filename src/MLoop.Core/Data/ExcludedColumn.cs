using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// A column featurization drops, with the reason it was dropped.
/// </summary>
/// <param name="Name">Column name as it appears in the CSV header.</param>
/// <param name="Reason">
/// One of <see cref="SchemaDataTypes.ExcludedDateTime"/>, <see cref="SchemaDataTypes.ExcludedSparse"/>,
/// or <see cref="SchemaDataTypes.ExcludedConstant"/> — the same vocabulary the saved schema records,
/// so the decision survives into predict and evaluate unchanged.
/// </param>
/// <seealso cref="CsvDataLoader.DetermineExcludedColumns"/>
public sealed record ExcludedColumn(string Name, string Reason);
