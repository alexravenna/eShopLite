﻿using Azure.Search.Documents.Indexes;
using DataEntities;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.AzureAISearch;
using Newtonsoft.Json;
using OpenAI.Chat;
using OpenAI.Embeddings;
using Products.Models;
using SearchEntities;
using System.Text;
using VectorEntities;

namespace Products.Memory;

public class MemoryContext
{
    private ILogger _logger;
    public ChatClient? _chatClient;
    public EmbeddingClient? _embeddingClient;
    public SearchIndexClient? _azureSearchIndexClient;
    // Collection for storing and searching product vectors in the memory context
    public VectorStoreCollection<string, ProductVector> _productsCollection;
    private string _systemPrompt = "";
    private bool _isMemoryCollectionInitialized = false;

    public MemoryContext(ILogger logger, ChatClient? chatClient, EmbeddingClient? embeddingClient, SearchIndexClient? azureSearchIndexClient)
    {
        _logger = logger;
        _chatClient = chatClient;
        _embeddingClient = embeddingClient;
        _azureSearchIndexClient = azureSearchIndexClient;

        _logger.LogInformation("Memory context created");
        _logger.LogInformation($"Chat Client is null: {_chatClient is null}");
        _logger.LogInformation($"Embedding Client is null: {_embeddingClient is null}");
        _logger.LogInformation($"Azure Search Index Client  is null: {_azureSearchIndexClient is null}");
    }

    public async Task<bool> InitMemoryContextAsync(Context db)
    {
        _logger.LogInformation("Initializing memory context with Azure AI Search");

        var vectorProductStore = new AzureAISearchVectorStore(_azureSearchIndexClient);
        _productsCollection = vectorProductStore.GetCollection<string, ProductVector>("products");
        await _productsCollection.EnsureCollectionExistsAsync();

        // define system prompt
        _systemPrompt = "You are a useful assistant. You always reply with a short and funny message. If you do not know an answer, you say 'I don't know that.' You only answer questions related to outdoor camping products. For any other type of questions, explain to the user that you only answer outdoor camping products questions. Do not store memory of the chat conversation.";

        _logger.LogInformation("Get a copy of the list of products");
        // get a copy of the list of products
        var products = await db.Product.ToListAsync();

        _logger.LogInformation("Filling products in memory");

        // iterate over the products and add them to the memory
        foreach (var product in products)
        {
            try
            {
                _logger.LogInformation("Adding product to memory: {Product}", product.Name);
                var productInfo = $"[{product.Name}] is a product that costs [{product.Price}] and is described as [{product.Description}]";

                var result = await _embeddingClient.GenerateEmbeddingAsync(productInfo);

                // new product vector
                var productVector = new ProductVector
                {
                    Id = product.Id.ToString(), // Convert int to string explicitly
                    Name = product.Name,
                    Description = product.Description,
                    Price = product.Price.ToString(),
                    ImageUrl = product.ImageUrl,
                    Vector = result.Value.ToFloats()
                };

                await _productsCollection.UpsertAsync(productVector);
                _logger.LogInformation("Product added to memory: {Product}", product.Name);
            }
            catch (Exception exc)
            {
                _logger.LogError(exc, "Error adding product to memory");
            }
        }

        _logger.LogInformation("DONE! Filling products in memory");
        return true;
    }

    public async Task<SearchResponse> Search(string search, Context db)
    {
        if (!_isMemoryCollectionInitialized)
        {
            await InitMemoryContextAsync(db);
            _isMemoryCollectionInitialized = true;
        }

        var response = new SearchResponse
        {
            Response = $"I don't know the answer for your question. Your question is: [{search}]"
        };

        try
        {
            var result = await _embeddingClient.GenerateEmbeddingAsync(search);
            var vectorSearchQuery = result.Value.ToFloats();

            // search the vector database for the most similar product        
            var sbFoundProducts = new StringBuilder();
            int productPosition = 1;

            await foreach (var resultItem in _productsCollection.SearchAsync(vectorSearchQuery, top: 3))
            {
                if (resultItem.Score > 0.5)
                {
                    int prodId = int.Parse(resultItem.Record.Id);   
                    var product = await db.FindAsync<Product>(prodId);
                    if (product != null)
                    {
                        response.Products.Add(product);
                        sbFoundProducts.AppendLine($"- Product {productPosition}:");
                        sbFoundProducts.AppendLine($"  - Name: {product.Name}");
                        sbFoundProducts.AppendLine($"  - Description: {product.Description}");
                        sbFoundProducts.AppendLine($"  - Price: {product.Price}");
                        productPosition++;
                    }
                }
            }

            // let's improve the response message
            var prompt = @$"You are an intelligent assistant helping clients with their search about outdoor products. 
Generate a catchy and friendly message using the information below.
Add a comparison between the products found and the search criteria.
Include products details.
    - User Question: {search}
    - Found Products: 
{sbFoundProducts}";

            var messages = new List<ChatMessage>
    {
        new SystemChatMessage(_systemPrompt),
        new UserChatMessage(prompt)
    };

            _logger.LogInformation("{ChatHistory}", JsonConvert.SerializeObject(messages));

            var resultPrompt = await _chatClient.CompleteChatAsync(messages);
            response.Response = resultPrompt.Value.Content[0].Text!;

        }
        catch (Exception ex)
        {
            response.Response = $"An error occurred: {ex.Message}";
            _logger.LogError(ex, "Error during search");
        }
        return response;
    }

    //    public async Task<SearchResponse> Search(string search, Context db)
    //    {
    //        if (!_isMemoryCollectionInitialized)
    //        {
    //            await InitMemoryContextAsync(db);
    //            _isMemoryCollectionInitialized = true;
    //        }

    //        var response = new SearchResponse();
    //        response.Response = $"I don't know the answer for your question. Your question is: [{search}]";
    //        Product? firstProduct = new Product();
    //        var responseText = "";
    //        try
    //        {
    //            var result = await _embeddingClient.GenerateEmbeddingAsync(search);
    //            var vectorSearchQuery = result.Value.ToFloats();

    //            var searchOptions = new VectorSearchOptions<ProductVectorAzureAISearch>()
    //            {
    //                Top = 3
    //            };

    //            // search the vector database for the most similar product        
    //            var searchResults = await _productsCollection.VectorizedSearchAsync(vectorSearchQuery, searchOptions);
    //            double searchScore = 0.0;
    //            await foreach (var searchItem in searchResults.Results)
    //            {
    //                if (searchItem.Score > 0.5)
    //                {
    //                    // product found, search the db for the product details                    
    //                    firstProduct = new Product
    //                    {
    //                        Id = int.Parse(searchItem.Record.Id),
    //                        Name = searchItem.Record.Name,
    //                        Description = searchItem.Record.Description,
    //                        Price = decimal.Parse(searchItem.Record.Price),
    //                        ImageUrl = searchItem.Record.ImageUrl
    //                    };

    //                    searchScore = searchItem.Score.Value;
    //                    responseText = $"The product [{firstProduct.Name}] fits with the search criteria [{search}][{searchItem.Score.Value.ToString("0.00")}]";
    //                    _logger.LogInformation($"Search Response: {responseText}");
    //                }
    //            }

    //            // let's improve the response message
    //            var prompt = @$"You are an intelligent assistant helping clients with their search about outdoor products. Generate a catchy and friendly message using the following information:
    //    - User Question: {search}
    //    - Found Product Name: {firstProduct.Name}
    //    - Found Product Description: {firstProduct.Description}
    //    - Found Product Price: {firstProduct.Price}
    //Include the found product information in the response to the user question.";

    //            var messages = new List<ChatMessage>
    //    {
    //        new SystemChatMessage(_systemPrompt),
    //        new UserChatMessage(prompt)
    //    };

    //            _logger.LogInformation("{ChatHistory}", JsonConvert.SerializeObject(messages));

    //            var resultPrompt = await _chatClient.CompleteChatAsync(messages);
    //            responseText = resultPrompt.Value.Content[0].Text!;

    //            // create a response object
    //            response = new SearchResponse
    //            {
    //                Products = firstProduct == null ? [new Product()] : [firstProduct],
    //                Response = responseText
    //            };

    //        }
    //        catch (Exception ex)
    //        {
    //            // Handle exceptions (log them, rethrow, etc.)
    //            response.Response = $"An error occurred: {ex.Message}";
    //        }
    //        return response;
    //    }
}