using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Hooks;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Hooks;

public class HookEngineTests : IDisposable
{
    private readonly string _tempProjectRoot;
    private readonly TestLogger _logger;
    private readonly HookEngine _engine;
    private readonly MLContext _mlContext;

    public HookEngineTests()
    {
        _tempProjectRoot = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempProjectRoot);
        _logger = new TestLogger();
        _engine = new HookEngine(_tempProjectRoot, _logger);
        _mlContext = new MLContext(seed: 42);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempProjectRoot))
        {
            Directory.Delete(_tempProjectRoot, recursive: true);
        }
    }

    [Fact]
    public void GetHookDirectory_ReturnsCorrectPath_ForPreTrain()
    {
        // Act
        var path = _engine.GetHookDirectory(HookType.PreTrain);

        // Assert
        var expected = Path.Combine(_tempProjectRoot, ".mloop", "scripts", "hooks", "pre-train");
        Assert.Equal(expected, path);
    }

    [Fact]
    public void GetHookDirectory_ReturnsCorrectPath_ForPostTrain()
    {
        // Act
        var path = _engine.GetHookDirectory(HookType.PostTrain);

        // Assert
        var expected = Path.Combine(_tempProjectRoot, ".mloop", "scripts", "hooks", "post-train");
        Assert.Equal(expected, path);
    }

    [Fact]
    public void GetHookDirectory_ReturnsCorrectPath_ForPrePredict()
    {
        // Act
        var path = _engine.GetHookDirectory(HookType.PrePredict);

        // Assert
        var expected = Path.Combine(_tempProjectRoot, ".mloop", "scripts", "hooks", "pre-predict");
        Assert.Equal(expected, path);
    }

    [Fact]
    public void GetHookDirectory_ReturnsCorrectPath_ForPostEvaluate()
    {
        // Act
        var path = _engine.GetHookDirectory(HookType.PostEvaluate);

        // Assert
        var expected = Path.Combine(_tempProjectRoot, ".mloop", "scripts", "hooks", "post-evaluate");
        Assert.Equal(expected, path);
    }

    [Fact]
    public void InitializeDirectories_CreatesAllHookDirectories()
    {
        // Act
        _engine.InitializeDirectories();

        // Assert
        Assert.True(Directory.Exists(_engine.GetHookDirectory(HookType.PreTrain)));
        Assert.True(Directory.Exists(_engine.GetHookDirectory(HookType.PostTrain)));
        Assert.True(Directory.Exists(_engine.GetHookDirectory(HookType.PrePredict)));
        Assert.True(Directory.Exists(_engine.GetHookDirectory(HookType.PostEvaluate)));
    }

    [Fact]
    public void HasHooks_WithNoDirectory_ReturnsFalse()
    {
        // Act
        var hasHooks = _engine.HasHooks(HookType.PreTrain);

        // Assert
        Assert.False(hasHooks);
    }

    [Fact]
    public void HasHooks_WithEmptyDirectory_ReturnsFalse()
    {
        // Arrange
        _engine.InitializeDirectories();

        // Act
        var hasHooks = _engine.HasHooks(HookType.PreTrain);

        // Assert
        Assert.False(hasHooks);
    }

    [Fact]
    public void HasHooks_WithScripts_ReturnsTrue()
    {
        // Arrange
        _engine.InitializeDirectories();
        var scriptPath = Path.Combine(_engine.GetHookDirectory(HookType.PreTrain), "01_test.cs");
        File.WriteAllText(scriptPath, "// test hook");

        // Act
        var hasHooks = _engine.HasHooks(HookType.PreTrain);

        // Assert
        Assert.True(hasHooks);
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithNoDirectory_ReturnsTrue()
    {
        // Arrange
        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithEmptyDirectory_ReturnsTrue()
    {
        // Arrange
        _engine.InitializeDirectories();
        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithContinueHook_ReturnsTrue()
    {
        // Arrange
        _engine.InitializeDirectories();
        var hookPath = Path.Combine(_engine.GetHookDirectory(HookType.PreTrain), "01_continue.cs");

        var hookScript = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestContinueHook : IMLoopHook
{
    public string Name => ""Test Continue Hook"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        context.Logger.Info(""Hook executed successfully"");
        return Task.FromResult(HookResult.Continue(""Processing complete""));
    }
}
";
        File.WriteAllText(hookPath, hookScript);
        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.True(result);
        Assert.Contains("Hook executed successfully", _logger.InfoMessages);
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithAbortHook_ReturnsFalse()
    {
        // Arrange
        _engine.InitializeDirectories();
        var hookPath = Path.Combine(_engine.GetHookDirectory(HookType.PreTrain), "01_abort.cs");

        var hookScript = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestAbortHook : IMLoopHook
{
    public string Name => ""Test Abort Hook"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        return Task.FromResult(HookResult.Abort(""Data quality insufficient""));
    }
}
";
        File.WriteAllText(hookPath, hookScript);
        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.False(result);
        Assert.Contains(_logger.ErrorMessages, msg => msg.Contains("Data quality insufficient"));
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithModifyConfigHook_ModifiesMetadata()
    {
        // Arrange
        _engine.InitializeDirectories();
        var hookPath = Path.Combine(_engine.GetHookDirectory(HookType.PreTrain), "01_modify.cs");

        var hookScript = @"
using System.Threading.Tasks;
using System.Collections.Generic;
using MLoop.Extensibility.Hooks;

public class TestModifyConfigHook : IMLoopHook
{
    public string Name => ""Test Modify Config Hook"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        var modifications = new Dictionary<string, object>
        {
            [""CustomSetting""] = ""CustomValue""
        };
        return Task.FromResult(HookResult.ModifyConfig(modifications, ""Config modified""));
    }
}
";
        File.WriteAllText(hookPath, hookScript);
        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.True(result);
        Assert.Equal("CustomValue", context.Metadata["CustomSetting"]);
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithMultipleHooks_ExecutesInAlphabeticalOrder()
    {
        // Arrange
        _engine.InitializeDirectories();
        var hookDir = _engine.GetHookDirectory(HookType.PreTrain);

        var hookScript1 = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestHook1 : IMLoopHook
{
    public string Name => ""Hook 1"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        context.Logger.Info(""Hook 1 executed"");
        return Task.FromResult(HookResult.Continue());
    }
}
";
        var hookScript2 = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestHook2 : IMLoopHook
{
    public string Name => ""Hook 2"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        context.Logger.Info(""Hook 2 executed"");
        return Task.FromResult(HookResult.Continue());
    }
}
";
        File.WriteAllText(Path.Combine(hookDir, "02_hook2.cs"), hookScript2);
        File.WriteAllText(Path.Combine(hookDir, "01_hook1.cs"), hookScript1);

        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.True(result);
        var hook1Index = _logger.InfoMessages.FindIndex(m => m.Contains("Hook 1 executed"));
        var hook2Index = _logger.InfoMessages.FindIndex(m => m.Contains("Hook 2 executed"));
        Assert.True(hook1Index < hook2Index, "Hooks should execute in alphabetical order");
    }

    [Fact]
    public async Task ExecuteHooksAsync_WithFailingHook_ContinuesExecution()
    {
        // Arrange
        _engine.InitializeDirectories();
        var hookPath = Path.Combine(_engine.GetHookDirectory(HookType.PreTrain), "01_failing.cs");

        var hookScript = @"
using System;
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestFailingHook : IMLoopHook
{
    public string Name => ""Test Failing Hook"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        throw new InvalidOperationException(""Hook execution failed"");
    }
}
";
        File.WriteAllText(hookPath, hookScript);
        var context = CreateTestContext(HookType.PreTrain);

        // Act
        var result = await _engine.ExecuteHooksAsync(HookType.PreTrain, context);

        // Assert
        Assert.True(result); // Should continue despite hook failure
        Assert.Contains(_logger.ErrorMessages, msg => msg.Contains("Hook execution failed"));
    }

    private HookContext CreateTestContext(HookType hookType)
    {
        // Create minimal test data
        var data = new[]
        {
            new TestData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
            new TestData { Feature1 = 3.0f, Feature2 = 4.0f, Label = false }
        };
        var dataView = _mlContext.Data.LoadFromEnumerable(data);

        return new HookContext
        {
            HookType = hookType,
            HookName = "test-hook",
            MLContext = _mlContext,
            DataView = dataView,
            Model = null,
            ExperimentResult = null,
            Metrics = null,
            ProjectRoot = _tempProjectRoot,
            Logger = _logger,
            Metadata = new Dictionary<string, object>
            {
                ["LabelColumn"] = "Label",
                ["TaskType"] = "BinaryClassification",
                ["ModelName"] = "test-model"
            }
        };
    }

    private class TestLogger : ILogger
    {
        public List<string> InfoMessages { get; } = new();
        public List<string> WarningMessages { get; } = new();
        public List<string> ErrorMessages { get; } = new();
        public List<string> DebugMessages { get; } = new();

        public void Info(string message) => InfoMessages.Add(message);
        public void Warning(string message) => WarningMessages.Add(message);
        public void Error(string message) => ErrorMessages.Add(message);
        public void Error(string message, Exception exception) => ErrorMessages.Add($"{message}{Environment.NewLine}{exception}");
        public void Debug(string message) => DebugMessages.Add(message);
    }

    private class TestData
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public bool Label { get; set; }
    }
}
