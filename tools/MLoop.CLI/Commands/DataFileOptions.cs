using System.CommandLine;

namespace MLoop.CLI.Commands;

/// <summary>Options that describe the data file a command is given, shared so each means one thing everywhere.</summary>
internal static class DataFileOptions
{
    /// <summary>
    /// <c>--records</c>: for a JSON data file whose rows sit in a nested array, the dotted path to that
    /// array. How the rows are read — and which enclosing fields they carry — is FilePrepper's rule
    /// (<c>JsonUtils.ReadJsonFileAsync</c>), not restated here.
    /// </summary>
    public static Option<string?> Records() => new("--records")
    {
        Description = "For a JSON data file: the dotted path to the array whose items are the rows, "
            + "e.g. data.paragraphs.qas.answers. Each row also carries the fields of the items it sits in, "
            + "named by their array (paragraphs.context)."
    };
}
