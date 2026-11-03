using System.Text;
using MLoop.Tests.Commands;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The usual Korean spreadsheet export - CP949, Korean column names, Korean class words, a dataframe's index in front, a header cell broken
/// over two lines - is a table every command must read as the table it holds. Reading the first line of a file as UTF-8 is a one-line habit that works for every table with ASCII names, so a label that is
/// not ASCII was refused as "not found" in a CP949 file (a plain-ASCII label hid it). Each command of
/// <see cref="EveryCommandRunsContractTests.Runs"/> runs once more in a project trained on such tables, with the columns it names spelled
/// the way the files spell them, and must show the same effect.
/// </summary>
[Collection("FileSystem")]
public sealed class EveryCommandReadsAKoreanCp949TableTests : IClassFixture<EveryCommandReadsAKoreanCp949TableTests.KoreanProject>
{
    private readonly KoreanProject _project;

    public EveryCommandReadsAKoreanCp949TableTests(KoreanProject project) => _project = project;

    /// <summary>The project of the contract test, trained on files written in CP949 with Korean column names and class words.</summary>
    public sealed class KoreanProject : EveryCommandRunsContractTests.TrainedProject
    {
        public override string[] Names => ["엑스", "제트", "더블유", "결과"];

        // The way these exports are shaped: a dataframe's index in front (empty name), and a header cell broken over two lines.
        public override string[] HeaderCells => ["", "엑스", "제트", "\"더블\n유\"", "결과"];

        public override string[] ReadNames => ["엑스", "제트", "더블 유", "결과"];

        public override string RowPrefix(int row) => $"{row},";

        public override string[] Classes => ["예", "아니오"];

        public override Encoding FileEncoding
        {
            get
            {
                Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
                return Encoding.GetEncoding(949);
            }
        }

        public override bool Workbooks => false;
    }

    /// <summary>The command line with the columns it names spelled as the project's files spell them.</summary>
    internal static string[] Spelled(string[] args, string[] names) =>
    [
        .. args.Select((a, i) => i > 0 && args[i - 1] is "--label" or "--column" or "--columns" or "--drop"
            ? a switch { "x" => names[0], "z" => names[1], "w" => names[2], "y" => names[3], _ => a }
            : a)
    ];

    public static TheoryData<string> RunNames()
    {
        var data = new TheoryData<string>();
        foreach (var name in EveryCommandRunsContractTests.Runs.Keys.Order(StringComparer.Ordinal)) data.Add(name);
        return data;
    }

    [Fact]
    public void TheSpelling_NamesTheColumnsAndLeavesTheRestAlone()
    {
        var names = new[] { "가", "나", "다", "라" };
        Assert.Equal(["analyze", "importance", "datasets/train.csv", "--label", "라", "--json"],
            Spelled(["analyze", "importance", "datasets/train.csv", "--label", "y", "--json"], names));
        Assert.Equal(["detect", "datasets/train.csv", "--column", "가", "--json"], Spelled(["detect", "datasets/train.csv", "--column", "x", "--json"], names));
        Assert.Equal(["features", "select", "--drop", "나", "--json"], Spelled(["features", "select", "--drop", "z", "--json"], names));
        // a word that is not after a column option is not a column: "x" of an experiment id or a value stays
        Assert.Equal(["threshold", "exp-001", "--precision", "0.8"], Spelled(["threshold", "exp-001", "--precision", "0.8"], names));
        // the cases that name a column are among those run, so the spelling is exercised
        Assert.Equal(4, EveryCommandRunsContractTests.Runs.Values.Count(r => !Spelled(r.Args, names).SequenceEqual(r.Args)));
    }

    [Theory]
    [MemberData(nameof(RunNames))]
    public async Task TheCommandRunsOnAKoreanCp949TableAndShowsItsEffect(string name)
    {
        var plan = EveryCommandRunsContractTests.Runs[name];
        var args = Spelled(plan.Args, _project.Names);

        var seen = await _project.RunAsync(args);

        Assert.True(seen.Exit == plan.Exit, $"mloop {string.Join(' ', args)} exited {seen.Exit}\n{seen.Out}\n{seen.Err}");
        if (name is "info" or "analyze profile")
        {
            // the names come back as the file spells them, not as replacement characters
            Assert.Equal(_project.ReadNames, ColumnNamesOf(name, seen));
        }
        else
        {
            plan.Effect(seen, _project.Root);
        }
        if (plan.Then is { } undo) await _project.RunAsync(undo);
    }

    private static IEnumerable<string> ColumnNamesOf(string command, EveryCommandRunsContractTests.Seen seen)
    {
        var json = seen.Json();
        var columns = command == "info" ? json.GetProperty("columns") : json.GetProperty("data").GetProperty("columns");
        return columns.EnumerateArray().Select(c => c.GetProperty("name").GetString()!);
    }
}
