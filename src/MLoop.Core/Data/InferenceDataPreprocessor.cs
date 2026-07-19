using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// The single, shared CSV preprocessing sequence applied at inference time by both predict and
/// evaluate. It mirrors the training path's column-shaping (encoding → flatten → index → exclude /
/// fallback) so the inference feature vector reproduces exactly what the model was fitted on.
///
/// This centralizes logic that previously diverged across <c>PredictionEngine</c> and
/// <c>EvaluationEngine</c> — the root cause of BUG-43 (CP949 encoding garbled in predict's manual BOM
/// check) and BUG-44 (index/exclude columns left in, widening the feature vector), plus the silent
/// gaps that divergence hid: multiline flattening and constant-column removal were missing from both
/// inference paths.
///
/// Training (<see cref="CsvDataLoader.LoadData"/>) remains the source of truth: it *discovers*
/// DateTime/sparse/constant columns from the data and records them as "Exclude" in the captured
/// schema. Inference is the *applier* — when a trained schema is supplied, exclusion is deterministic
/// (driven by that record, not the inference data); without a schema it falls back to the same
/// data-dependent removals training would have run.
/// </summary>
public static class InferenceDataPreprocessor
{
    /// <summary>
    /// Runs the inference preprocessing sequence over <paramref name="inputPath"/> and returns the
    /// path to feed into the model. Every intermediate file the sequence creates is reported in
    /// <paramref name="tempFiles"/> (including the returned path when it is itself a temp); the caller
    /// must delete them after the lazy <c>IDataView</c> consumption completes. The original
    /// <paramref name="inputPath"/> is never modified or reported.
    /// </summary>
    public static string Prepare(
        string inputPath,
        string? labelColumn,
        InputSchemaInfo? trainedSchema,
        out List<string> tempFiles,
        Action<string>? log = null)
    {
        tempFiles = new List<string>();
        string current = inputPath;

        // 1. Encoding: CP949/EUC-KR → UTF-8 with BOM (same detector CsvDataLoader.EnsureUtf8Bom uses).
        //    Replaces predict's hand-rolled BOM check, which read non-UTF-8 files as UTF-8 (BUG-43).
        var (utf8Path, detection) = EncodingDetector.ConvertToUtf8WithBom(current);
        if (detection.WasConverted)
        {
            tempFiles.Add(utf8Path);
            current = utf8Path;
        }

        // 2. Flatten multiline quoted fields and headers (ML.NET's TextLoader has no RFC 4180
        //    multiline support). Was present in training but missing from both inference paths.
        current = Step(CsvDataLoader.FlattenMultiLineQuotedFields(current, log), current, tempFiles);
        current = Step(CsvDataLoader.FlattenMultiLineHeaders(current), current, tempFiles);

        // 3. Index columns (pandas index / "Unnamed: N"). Data-independent, safe at any size (BUG-44).
        current = Step(CsvDataLoader.RemoveIndexColumns(current, log), current, tempFiles);

        // 4. Column exclusion. With a trained schema this is deterministic — applying the DateTime /
        //    constant / sparse columns training marked "Exclude". Without one, fall back to the same
        //    data-dependent removals training runs, so predict and evaluate stay identical.
        if (trainedSchema != null)
        {
            var excluded = trainedSchema.Columns
                .Where(c => c.Purpose == "Exclude")
                .Select(c => c.Name);
            current = Step(CsvDataLoader.RemoveExcludedColumns(current, excluded, log), current, tempFiles);
        }
        else
        {
            // No recorded decision to apply, so the training-time one is reconstructed from this file.
            // That reconstruction is a guess: the rules are data-dependent, and inference data is a
            // different slice than training data, so a column that was constant in one and not the
            // other yields a feature vector the model was not fitted on. Unlike the training path —
            // where the decision exists and is now shared (CsvDataLoader.DetermineExcludedColumns) —
            // here the information is genuinely absent, so the fallback stays; what it must not do is
            // stay silent, because the symptom it produces is an opaque ML.NET schema-mismatch error.
            var before = ReadHeaderCount(current);
            current = Step(CsvDataLoader.RemoveDateTimeColumns(current, labelColumn, log), current, tempFiles);
            current = Step(CsvDataLoader.RemoveSparseColumns(current, labelColumn, log: log), current, tempFiles);
            current = Step(CsvDataLoader.RemoveConstantColumns(current, labelColumn, log), current, tempFiles);

            if (ReadHeaderCount(current) != before)
            {
                (log ?? Console.WriteLine)(
                    "[Warning] This model has no recorded feature-exclusion schema, so the excluded columns " +
                    "were re-derived from the inference data. If the result disagrees with what training " +
                    "dropped, the model will report a feature-vector size mismatch — retrain to record the " +
                    "schema (mloop train).");
            }
        }

        return current;
    }

    /// <summary>
    /// Column count of the CSV header, or -1 when the file cannot be read (the caller only compares
    /// two readings of the same file, so an unreadable file simply reports "unchanged").
    /// </summary>
    private static int ReadHeaderCount(string path)
    {
        try
        {
            using var reader = new StreamReader(path, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            var header = reader.ReadLine();
            return header is null ? -1 : Prediction.CsvFieldParser.ParseFields(header).Length;
        }
        catch
        {
            return -1;
        }
    }

    /// <summary>
    /// Records a freshly created temp file. Each Remove*/Flatten* helper returns its input unchanged
    /// when it makes no change, or a new UTF-8 BOM temp when it does — so a changed path is a temp.
    /// </summary>
    private static string Step(string result, string previous, List<string> tempFiles)
    {
        if (!string.Equals(result, previous, StringComparison.Ordinal))
            tempFiles.Add(result);
        return result;
    }
}
