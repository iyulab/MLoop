using System.CommandLine;

using Microsoft.Extensions.Logging;

using FilePrepper.CLI.Commands;

namespace MLoop.CLI.Commands;

/// <summary>
/// Data preprocessing command - integrates FilePrepper CLI commands
/// Usage: mloop prep [command] [options]
/// </summary>
public class PrepCommand : Command
{
    private readonly ILoggerFactory _loggerFactory;

    public PrepCommand() : base("prep", "Data preprocessing tools (powered by FilePrepper)")
    {
        // Create a minimal logger factory for FilePrepper commands
        _loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.SetMinimumLevel(LogLevel.Warning);
        });

        // Add all FilePrepper commands as subcommands
        AddFilePrepperCommands();
    }

    private void AddFilePrepperCommands()
    {
        // Data Transformation Commands
        this.Add(new AddColumnsCommand(_loggerFactory));
        this.Add(new RemoveColumnsCommand(_loggerFactory));
        this.Add(new RenameColumnsCommand(_loggerFactory));
        this.Add(new ReorderColumnsCommand(_loggerFactory));
        this.Add(new DataTypeConvertCommand(_loggerFactory));

        // Data Cleaning Commands
        this.Add(new FillMissingValuesCommand(_loggerFactory));
        this.Add(new DropDuplicatesCommand(_loggerFactory));
        this.Add(new FilterRowsCommand(_loggerFactory));
        this.Add(new CSVCleanerCommand(_loggerFactory));
        this.Add(new ValueReplaceCommand(_loggerFactory));
        this.Add(new RemoveConstantsCommand(_loggerFactory));

        // Feature Engineering Commands
        this.Add(new CreateLagFeaturesCommand(_loggerFactory));
        this.Add(new OneHotEncodingCommand(_loggerFactory));
        this.Add(new NormalizeDataCommand(_loggerFactory));
        this.Add(new ScaleDataCommand(_loggerFactory));
        this.Add(new ColumnInteractionCommand(_loggerFactory));

        // Date/Time Commands
        this.Add(new DateExtractionCommand(_loggerFactory));
        this.Add(new DateTimeCommand(_loggerFactory));

        // Aggregation & Statistics Commands
        this.Add(new AggregateCommand(_loggerFactory));
        this.Add(new BasicStatisticsCommand(_loggerFactory));
        this.Add(new WindowCommand(_loggerFactory));

        // Data Merging Commands
        this.Add(new MergeCommand(_loggerFactory));
        this.Add(new MergeAsOfCommand(_loggerFactory));

        // Expression & Conditional Commands
        this.Add(new ExpressionCommand(_loggerFactory));
        this.Add(new ConditionalCommand(_loggerFactory));

        // String Operations
        this.Add(new StringCommand(_loggerFactory));

        // Data Sampling & Reshaping
        this.Add(new DataSamplingCommand(_loggerFactory));
        this.Add(new UnpivotCommand(_loggerFactory));

        // Format Conversion
        this.Add(new FileFormatConvertCommand(_loggerFactory));
    }
}
