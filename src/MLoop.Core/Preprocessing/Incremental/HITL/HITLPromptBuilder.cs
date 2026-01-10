using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.HITL.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using Spectre.Console;
using System.Diagnostics;

namespace MLoop.Core.Preprocessing.Incremental.HITL;

/// <summary>
/// Builds interactive prompts and collects user responses using Spectre.Console.
/// </summary>
public sealed class HITLPromptBuilder : IHITLPromptBuilder
{
    private readonly ILogger<HITLPromptBuilder> _logger;

    public HITLPromptBuilder(ILogger<HITLPromptBuilder> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc />
    public string BuildPrompt(HITLQuestion question)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[bold cyan]🤔 Human Decision Required[/]").LeftJustified());
        AnsiConsole.WriteLine();

        // Display context
        var contextPanel = new Panel(question.Context)
        {
            Header = new PanelHeader("[yellow]📊 Context[/]", Justify.Left),
            Border = BoxBorder.Rounded,
            BorderStyle = new Style(Color.Yellow)
        };
        AnsiConsole.Write(contextPanel);
        AnsiConsole.WriteLine();

        // Display main question
        AnsiConsole.MarkupLine($"[bold white]{question.Question}[/]");
        AnsiConsole.WriteLine();

        // Display options
        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .AddColumn(new TableColumn("[bold]Option[/]").Centered())
            .AddColumn(new TableColumn("[bold]Choice[/]"))
            .AddColumn(new TableColumn("[bold]Details[/]"));

        foreach (var option in question.Options)
        {
            var keyStyle = option.IsRecommended ? "[green bold]" : "[white]";
            var labelStyle = option.IsRecommended ? "[green]" : "[grey]";
            var recommendedBadge = option.IsRecommended ? " ⭐" : "";

            table.AddRow(
                $"{keyStyle}{option.Key}{recommendedBadge}[/]",
                $"{labelStyle}{option.Label}[/]",
                $"[dim]{option.Description}[/]"
            );
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        // Display recommendation if available
        if (!string.IsNullOrEmpty(question.RecommendedOption) &&
            !string.IsNullOrEmpty(question.RecommendationReason))
        {
            var recommendationPanel = new Panel(
                $"[green]Recommended:[/] Option [bold green]{question.RecommendedOption}[/]\n" +
                $"[dim]{question.RecommendationReason}[/]")
            {
                Header = new PanelHeader("[green]💡 AI Recommendation[/]", Justify.Left),
                Border = BoxBorder.Rounded,
                BorderStyle = new Style(Color.Green)
            };
            AnsiConsole.Write(recommendationPanel);
            AnsiConsole.WriteLine();
        }

        _logger.LogInformation("Displayed HITL question {QuestionId}", question.Id);
        return question.Id;
    }

    /// <inheritdoc />
    public HITLAnswer CollectAnswer(HITLQuestion question)
    {
        var stopwatch = Stopwatch.StartNew();

        string selectedOption;
        string? customValue = null;
        string? userRationale = null;

        // Collect option selection
        if (question.Type == HITLQuestionType.YesNo)
        {
            selectedOption = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[cyan]Your choice:[/]")
                    .AddChoices("Yes", "No")
            ) == "Yes" ? "Y" : "N";
        }
        else if (question.Type == HITLQuestionType.MultipleChoice)
        {
            var validOptions = question.Options.Select(o => o.Key).ToList();

            selectedOption = AnsiConsole.Prompt(
                new TextPrompt<string>("[cyan]Enter your choice (e.g., A, B, C):[/]")
                    .ValidationErrorMessage("[red]Invalid option. Please choose from the available options.[/]")
                    .Validate(input =>
                    {
                        var upper = input.Trim().ToUpperInvariant();
                        return validOptions.Contains(upper)
                            ? ValidationResult.Success()
                            : ValidationResult.Error($"Please enter one of: {string.Join(", ", validOptions)}");
                    })
            ).Trim().ToUpperInvariant();
        }
        else if (question.Type == HITLQuestionType.NumericInput)
        {
            selectedOption = "CUSTOM";
            customValue = AnsiConsole.Prompt(
                new TextPrompt<string>("[cyan]Enter numeric value:[/]")
                    .ValidationErrorMessage("[red]Please enter a valid number.[/]")
                    .Validate(input =>
                    {
                        return double.TryParse(input, out _)
                            ? ValidationResult.Success()
                            : ValidationResult.Error("Must be a valid number");
                    })
            );
        }
        else if (question.Type == HITLQuestionType.TextInput)
        {
            selectedOption = "CUSTOM";
            customValue = AnsiConsole.Prompt(
                new TextPrompt<string>("[cyan]Enter custom value:[/]")
                    .AllowEmpty()
            );
        }
        else // Confirmation
        {
            selectedOption = AnsiConsole.Confirm("[cyan]Do you approve this action?[/]") ? "APPROVE" : "REJECT";
        }

        // Optionally collect rationale
        if (AnsiConsole.Confirm("[dim]Would you like to add a note explaining your decision?[/]", defaultValue: false))
        {
            userRationale = AnsiConsole.Prompt(
                new TextPrompt<string>("[cyan]Your reasoning:[/]")
                    .AllowEmpty()
            );
        }

        stopwatch.Stop();

        var answer = new HITLAnswer
        {
            QuestionId = question.Id,
            SelectedOption = selectedOption,
            CustomValue = customValue,
            UserRationale = userRationale,
            TimeToDecide = stopwatch.Elapsed
        };

        _logger.LogInformation(
            "Collected answer for {QuestionId}: {Option} (took {Duration}ms)",
            question.Id, selectedOption, stopwatch.ElapsedMilliseconds);

        return answer;
    }

    /// <inheritdoc />
    public void DisplayConfirmation(HITLAnswer answer)
    {
        AnsiConsole.WriteLine();

        var confirmationText = $"✅ Decision recorded: [bold]{answer.SelectedOption}[/]";
        if (!string.IsNullOrEmpty(answer.CustomValue))
        {
            confirmationText += $"\n   Custom value: [cyan]{answer.CustomValue}[/]";
        }
        if (!string.IsNullOrEmpty(answer.UserRationale))
        {
            confirmationText += $"\n   Note: [dim]{answer.UserRationale}[/]";
        }
        confirmationText += $"\n   Decision time: [dim]{answer.TimeToDecide.TotalSeconds:F1}s[/]";

        var confirmationPanel = new Panel(confirmationText)
        {
            Border = BoxBorder.Rounded,
            BorderStyle = new Style(Color.Green)
        };

        AnsiConsole.Write(confirmationPanel);
        AnsiConsole.WriteLine();

        _logger.LogInformation("Displayed confirmation for {QuestionId}", answer.QuestionId);
    }
}
