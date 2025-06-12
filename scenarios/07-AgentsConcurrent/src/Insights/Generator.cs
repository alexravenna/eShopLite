using DataEntities;
using Insights.Models;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Orchestration;
using Microsoft.SemanticKernel.Agents.Orchestration.Concurrent;
using Microsoft.SemanticKernel.Agents.Orchestration.Transforms;
using Microsoft.SemanticKernel.Agents.Runtime.InProcess;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SqlServer.Server;
using OpenAI.Chat;

#pragma warning disable SKEXP0001, SKEXP0110 

namespace Insights;

public class Generator
{
    private ILogger _logger;
    public ChatClient? _chatClient;
    private Kernel _kernel;

    public Generator(ILogger logger, ChatClient? chatClient, Kernel kernel)
    {
        _logger = logger;
        _chatClient = chatClient;
        _kernel = kernel;
    }
    public async Task<string> GenerateInsightAsync(string search, Context db)
    {

        // Define the agents
        ChatCompletionAgent agentSentiment =
            CreateAgent(
                instructions: @"You are an expert in sentiment analysis. Given a string, evaluate its sentiment and return one of the following values: positive, neutral, or negative. 
The output should be in the format 'Sentiment:<detected sentiment>', in example: 'Sentiment:positive'",
                description: "An expert in evaluating and classifying the sentiment of text as positive, neutral, or negative.");
        ChatCompletionAgent agentLanguage =
            this.CreateAgent(
                instructions: @"You are an expert in language detection. Given a string, detect the language and return its standard language code (e.g., en for English, es for Spanish, fr for French, etc.).
The output should be in the format 'Language:<detected language>', in example: 'Language:en'",
                description: "An expert in detecting the language of text and returning its language code.");

        StructuredOutputTransform<Analysis> outputTransform = new(
            service: _chatClient.AsIChatClient().AsChatCompletionService(), 
            executionSettings: new OpenAIPromptExecutionSettings { ResponseFormat = typeof(Analysis) });
        ConcurrentOrchestration orchestration = new(agentSentiment, agentLanguage);

        // Start the runtime
        InProcessRuntime runtime = new();
        await runtime.StartAsync();
        OrchestrationResult<string[]> output = await orchestration.InvokeAsync(search, runtime);

        // analyze the result of the concurrent agents run
        string[] analysisResults = await output.GetValueAsync(TimeSpan.FromSeconds(60));
        var analysisResult = TransformToAnalysis(analysisResults);
        await runtime.RunUntilIdleAsync();

        // add insight to the database
        var insight = new UserQuestionInsight
        {
            CreatedAt = DateTime.UtcNow,
            Question = search,
            Sentiment = analysisResult.Sentiment, 
            Language = analysisResult.Language
        };
        db.UserQuestionInsight.Add(insight);
        await db.SaveChangesAsync();
        var sanitizedQuestion = insight.Question.Replace(Environment.NewLine, "").Replace("\n", "").Replace("\r", "");
        _logger.LogInformation($"Added insight: {sanitizedQuestion}");
        return "OK";
    }

    protected ChatCompletionAgent CreateAgent(string instructions, string? description = null, string? name = null)
    {
        return
            new ChatCompletionAgent
            {
                Name = name,
                Description = description,
                Instructions = instructions,
                Kernel = _kernel
            };
    }
    
    public Analysis TransformToAnalysis(string[] messages)
    {
        Analysis analysisResult = new Analysis();
        
        foreach (var message in messages)
        {
            // analyze the message to see if it is a sentiment or language, and set the result into the analysisResult object
            if (message.Contains("sentiment", StringComparison.OrdinalIgnoreCase))
            {
                // Extract sentiment from the message
                var sentiment = message.Split(':')[1].Trim();
                analysisResult.Sentiment = Enum.TryParse<Sentiment>(sentiment, true, out var parsedSentiment) ? parsedSentiment : Sentiment.NotDefined;
            }
            else if (message.Contains("language", StringComparison.OrdinalIgnoreCase))
            {
                // Extract language from the message
                var language = message.Split(':')[1].Trim();
                analysisResult.Language = language;
            }
        }

        return analysisResult;
    }
}

public class Analysis
{
    public Sentiment Sentiment { get; set; } = Sentiment.NotDefined;
    public string Language { get; set; } = string.Empty;
}