using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop init - Initializes a new MLoop project
/// </summary>
public static class InitCommand
{
    public static Command Create()
    {
        var projectNameArg = new Argument<string>("project-name")
        {
            Description = "Name of the project to create"
        };

        var taskOption = new Option<string>(
            "--task",
            getDefaultValue: () => "binary-classification",
            description: "ML task type: binary-classification, multiclass-classification, regression");

        var command = new Command("init", "Initialize a new ML project")
        {
            projectNameArg,
            taskOption
        };

        command.SetHandler(ExecuteAsync, projectNameArg, taskOption);

        return command;
    }

    private static async Task<int> ExecuteAsync(string projectName, string task)
    {
        try
        {
            AnsiConsole.MarkupLine($"[blue]Initializing MLoop project:[/] [green]{projectName}[/]");
            AnsiConsole.WriteLine();

            // Validate project name
            if (string.IsNullOrWhiteSpace(projectName))
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Project name cannot be empty");
                return 1;
            }

            if (projectName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Project name contains invalid characters");
                return 1;
            }

            // Validate task
            var validTasks = new[] { "binary-classification", "multiclass-classification", "regression" };
            if (!validTasks.Contains(task))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Invalid task type. Valid options: {string.Join(", ", validTasks)}");
                return 1;
            }

            // Check if directory already exists
            var projectPath = Path.Combine(Directory.GetCurrentDirectory(), projectName);
            if (Directory.Exists(projectPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Directory '{projectName}' already exists");
                return 1;
            }

            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var progressTask = ctx.AddTask("[green]Setting up project...[/]");

                    // Initialize filesystem and config
                    var fileSystem = new FileSystemManager();

                    // Step 1: Create directory structure
                    progressTask.Description = "[green]Creating directory structure...[/]";
                    await CreateDirectoryStructure(fileSystem, projectPath);
                    progressTask.Increment(30);

                    // Step 2: Create configuration files
                    progressTask.Description = "[green]Creating configuration files...[/]";
                    await CreateConfigurationFiles(fileSystem, projectPath, projectName, task);
                    progressTask.Increment(30);

                    // Step 3: Create documentation
                    progressTask.Description = "[green]Creating documentation...[/]";
                    await CreateDocumentation(fileSystem, projectPath, projectName, task);
                    progressTask.Increment(20);

                    // Step 4: Create .gitignore
                    progressTask.Description = "[green]Creating .gitignore...[/]";
                    await CreateGitIgnore(fileSystem, projectPath);
                    progressTask.Increment(20);

                    progressTask.Description = "[green]Project initialized![/]";
                });

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]✓[/] Project '{projectName}' created successfully!");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
            AnsiConsole.MarkupLine($"  1. cd {projectName}");
            AnsiConsole.MarkupLine("  2. Place your data in data/processed/train.csv");
            AnsiConsole.MarkupLine("  3. mloop train data/processed/train.csv --label <your-label-column>");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static async Task CreateDirectoryStructure(IFileSystemManager fileSystem, string projectPath)
    {
        // Create main project directory
        await fileSystem.CreateDirectoryAsync(projectPath);

        // Create .mloop directory (internal)
        var mloopPath = fileSystem.CombinePath(projectPath, ".mloop");
        await fileSystem.CreateDirectoryAsync(mloopPath);

        // Create data directories
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "data", "processed"));
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "data", "predictions"));

        // Create experiments directory
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "experiments"));

        // Create models directories
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "models", "staging"));
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "models", "production"));
    }

    private static async Task CreateConfigurationFiles(
        IFileSystemManager fileSystem,
        string projectPath,
        string projectName,
        string task)
    {
        // Create .mloop/config.json
        var config = new MLoopConfig
        {
            ProjectName = projectName,
            Task = task,
            Training = new TrainingSettings
            {
                TimeLimitSeconds = 300,
                Metric = "accuracy",
                TestSplit = 0.2
            }
        };

        var configData = new
        {
            project_name = projectName,
            version = "0.1.0",
            task,
            created_at = DateTime.UtcNow,
            mloop_version = "0.1.0-alpha"
        };

        var configPath = fileSystem.CombinePath(projectPath, ".mloop", "config.json");
        await fileSystem.WriteJsonAsync(configPath, configData);

        // Initialize experiment index
        var experimentIndex = new
        {
            next_id = 1,
            experiments = Array.Empty<object>()
        };

        var indexPath = fileSystem.CombinePath(projectPath, ".mloop", "experiment-index.json");
        await fileSystem.WriteJsonAsync(indexPath, experimentIndex);

        // Initialize model registry
        var registry = new Dictionary<string, object>();
        var registryPath = fileSystem.CombinePath(projectPath, ".mloop", "registry.json");
        await fileSystem.WriteJsonAsync(registryPath, registry);

        // Create mloop.yaml from template
        var yamlContent = GetYamlTemplate(projectName, task);
        var yamlPath = fileSystem.CombinePath(projectPath, "mloop.yaml");
        await fileSystem.WriteTextAsync(yamlPath, yamlContent);
    }

    private static async Task CreateDocumentation(
        IFileSystemManager fileSystem,
        string projectPath,
        string projectName,
        string task)
    {
        var readme = $@"# {projectName}

MLoop machine learning project for {task}.

## Quick Start

### 1. Prepare Your Data

Place your training data in `data/processed/train.csv`:

```bash
# Your CSV should have:
# - One column for the label (target variable)
# - Other columns as features
```

### 2. Train a Model

```bash
mloop train data/processed/train.csv --label <your-label-column>
```

### 3. Make Predictions

```bash
mloop predict experiments/exp-001/model.zip data/new-data.csv
```

## Project Structure

```
{projectName}/
├── .mloop/              # Internal MLoop metadata (gitignored)
├── mloop.yaml           # User configuration
├── data/
│   ├── processed/       # Your training/test data
│   └── predictions/     # Prediction outputs
├── experiments/         # Training experiments
├── models/
│   ├── staging/         # Staging models
│   └── production/      # Production models
└── README.md
```

## Configuration

Edit `mloop.yaml` to customize:
- Training time limit
- Evaluation metric
- Test split ratio

## Commands

- `mloop train` - Train a model
- `mloop predict` - Make predictions
- `mloop evaluate` - Evaluate model performance
- `mloop experiment list` - List all experiments

## Documentation

For more information, see [MLoop documentation](https://github.com/yourusername/mloop).
";

        var readmePath = fileSystem.CombinePath(projectPath, "README.md");
        await fileSystem.WriteTextAsync(readmePath, readme);
    }

    private static async Task CreateGitIgnore(IFileSystemManager fileSystem, string projectPath)
    {
        var gitignore = @"# MLoop user project .gitignore

# .NET Build outputs (if any scripts)
bin/
obj/
*.user
*.suo

# MLoop internal (keep metadata, ignore binaries)
.mloop/cache/

# Model binaries (large files)
experiments/*/model.zip
experiments/*/training.log
models/*/model.zip

# Prediction outputs
data/predictions/

# OS
.DS_Store
Thumbs.db
*.swp
";

        var gitignorePath = fileSystem.CombinePath(projectPath, ".gitignore");
        await fileSystem.WriteTextAsync(gitignorePath, gitignore);
    }

    private static string GetYamlTemplate(string projectName, string task)
    {
        var labelColumnExample = task switch
        {
            "binary-classification" => "Label",
            "multiclass-classification" => "Category",
            "regression" => "Target",
            _ => "Label"
        };

        return $@"# MLoop Project Configuration
project_name: {projectName}
task: {task}
label_column: {labelColumnExample}

# Training settings (optional - defaults provided)
training:
  time_limit_seconds: 300
  metric: accuracy
  test_split: 0.2

# Data paths (optional)
data:
  train: data/processed/train.csv
  test: data/processed/test.csv

# Model output (optional)
model:
  output_dir: models/staging
";
    }
}
