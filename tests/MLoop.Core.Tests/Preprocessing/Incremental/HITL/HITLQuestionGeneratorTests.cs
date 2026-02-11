using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.HITL;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using MLoop.Core.Tests.Preprocessing.Incremental.TestData;

namespace MLoop.Core.Tests.Preprocessing.Incremental.HITL;

public class HITLQuestionGeneratorTests
{
    [Fact]
    public void GenerateQuestion_WithMissingValueRule_GeneratesMultipleChoiceQuestion()
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rule = CreateMissingValueRule();
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act
        var question = generator.GenerateQuestion(rule, sample, analysis);

        // Assert
        Assert.NotNull(question);
        Assert.Equal(HITLQuestionType.MultipleChoice, question.Type);
        Assert.Contains("missing values", question.Question, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(4, question.Options.Count); // Delete, ImputeMean, ImputeMedian, ImputeCustom
        Assert.NotNull(question.RecommendedOption);
        Assert.NotNull(question.RecommendationReason);
        Assert.NotEmpty(question.RecommendationReason);
    }

    [Fact]
    public void GenerateQuestion_WithOutlierRule_GeneratesMultipleChoiceQuestion()
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rule = CreateOutlierRule();
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act
        var question = generator.GenerateQuestion(rule, sample, analysis);

        // Assert
        Assert.NotNull(question);
        Assert.Equal(HITLQuestionType.MultipleChoice, question.Type);
        Assert.Contains("outliers", question.Question, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(4, question.Options.Count); // Keep, Remove, Cap, Flag
        Assert.NotNull(question.RecommendedOption);
    }

    [Fact]
    public void GenerateQuestion_WithCategoryMappingRule_GeneratesMultipleChoiceQuestion()
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rule = CreateCategoryMappingRule();
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act
        var question = generator.GenerateQuestion(rule, sample, analysis);

        // Assert
        Assert.NotNull(question);
        Assert.Equal(HITLQuestionType.MultipleChoice, question.Type);
        Assert.Contains("category", question.Question, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(3, question.Options.Count);
        Assert.Equal("A", question.RecommendedOption); // Merge is default recommendation
    }

    [Fact]
    public void GenerateQuestion_WithNonHITLRule_ThrowsInvalidOperationException()
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rule = CreateNonHITLRule();
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act & Assert
        var exception = Assert.Throws<InvalidOperationException>(
            () => generator.GenerateQuestion(rule, sample, analysis)
        );
        Assert.Contains("does not require HITL", exception.Message);
    }

    [Fact]
    public void GenerateAllQuestions_WithMultipleHITLRules_GeneratesAllQuestions()
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rules = new List<PreprocessingRule>
        {
            CreateMissingValueRule(),
            CreateOutlierRule(),
            CreateNonHITLRule() // Should be skipped
        };
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act
        var questions = generator.GenerateAllQuestions(rules, sample, analysis);

        // Assert
        Assert.Equal(2, questions.Count); // Only HITL rules
        Assert.All(questions, q => Assert.NotNull(q.RecommendedOption));
    }

    [Fact]
    public void GenerateAllQuestions_WithFailingRule_SkipsAndContinues()
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rules = new List<PreprocessingRule>
        {
            CreateMissingValueRule(),
            CreateInvalidRule(), // Will fail
            CreateOutlierRule()
        };
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act
        var questions = generator.GenerateAllQuestions(rules, sample, analysis);

        // Assert
        Assert.Equal(2, questions.Count); // Invalid rule skipped
    }

    [Theory]
    [InlineData(0.03, "A")] // < 5% missing → Delete
    [InlineData(0.10, "B")] // > 5% missing → Impute mean
    public void GenerateQuestion_WithMissingValueRule_RecommendationBasedOnPercentage(
        double missingPercent,
        string expectedRecommendation)
    {
        // Arrange
        var generator = new HITLQuestionGenerator(NullLogger<HITLQuestionGenerator>.Instance);
        var rule = CreateMissingValueRule();
        rule.AffectedRows = (int)(100 * missingPercent);
        var sample = SampleDataGenerator.GenerateMixedData(100);
        var analysis = CreateMockAnalysis();

        // Act
        var question = generator.GenerateQuestion(rule, sample, analysis);

        // Assert
        Assert.Equal(expectedRecommendation, question.RecommendedOption);
    }

    // Helper methods
    private static PreprocessingRule CreateMissingValueRule()
    {
        return new PreprocessingRule
        {
            Id = "MissingValue_Age_Nulls",
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new List<string> { "Age" },
            Description = "Found NULL values in Age column",
            PatternType = PatternType.MissingValue,
            RequiresHITL = true,
            Priority = 5,
            AffectedRows = 10,
            Examples = new List<string> { "NULL", "N/A", "null" },
            DiscoveredInStage = 1
        };
    }

    private static PreprocessingRule CreateOutlierRule()
    {
        return new PreprocessingRule
        {
            Id = "Outlier_Salary_HighValues",
            Type = PreprocessingRuleType.OutlierHandling,
            ColumnNames = new List<string> { "Salary" },
            Description = "Detected outliers in Salary column",
            PatternType = PatternType.OutlierAnomaly,
            RequiresHITL = true,
            Priority = 4,
            AffectedRows = 5,
            Examples = new List<string> { "999999", "1000000" },
            DiscoveredInStage = 1
        };
    }

    private static PreprocessingRule CreateCategoryMappingRule()
    {
        return new PreprocessingRule
        {
            Id = "Category_Status_Variations",
            Type = PreprocessingRuleType.CategoryMapping,
            ColumnNames = new List<string> { "Status" },
            Description = "Found category variations in Status column",
            PatternType = PatternType.CategoryVariation,
            RequiresHITL = true,
            Priority = 3,
            AffectedRows = 15,
            Examples = new List<string> { "Active", "active", "ACTIVE" },
            DiscoveredInStage = 1
        };
    }

    private static PreprocessingRule CreateNonHITLRule()
    {
        return new PreprocessingRule
        {
            Id = "AutoFix_Whitespace_Trim",
            Type = PreprocessingRuleType.WhitespaceNormalization,
            ColumnNames = new List<string> { "Name" },
            Description = "Trim whitespace from Name column",
            PatternType = PatternType.WhitespaceIssue,
            RequiresHITL = false, // Auto-fixable
            Priority = 2,
            AffectedRows = 20,
            DiscoveredInStage = 1
        };
    }

    private static PreprocessingRule CreateInvalidRule()
    {
        return new PreprocessingRule
        {
            Id = "Invalid_Rule",
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new List<string> { "NonExistentColumn" }, // Invalid column
            Description = "Invalid rule for testing",
            PatternType = PatternType.MissingValue,
            RequiresHITL = true,
            Priority = 1,
            AffectedRows = 1,
            DiscoveredInStage = 1
        };
    }

    private static SampleAnalysis CreateMockAnalysis()
    {
        return new SampleAnalysis
        {
            StageNumber = 1,
            SampleRatio = 1.0,
            Timestamp = DateTime.UtcNow,
            RowCount = 100,
            ColumnCount = 5,
            QualityScore = 0.85,
            Columns = new List<ColumnAnalysis>()
        };
    }
}
