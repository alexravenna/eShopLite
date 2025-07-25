[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](/LICENSE)

## Description

**eShopLite - Realtime Audio Chat** is a reference .NET application implementing an eCommerce site with advanced search features and real-time audio capabilities. The solution combines keyword search, semantic search, and a new chat interface using audio powered by the **GPT-4o Realtime Audio API**. 

The reference application is part of the **[Generative AI for Beginners .NET](https://aka.ms/genainnet)** series, which aims to provide practical examples and resources for developers interested in generative AI.

- [Features](#features)
- [Architecture diagram](#architecture-diagram)
- [Getting started](#getting-started)
- [Deploying to Azure](#deploying)
- Run solution
  - [Run locally](#run-locally)
  - [Run the solution](#run-the-solution)
  - [Local development using existing models](#local-development-using-an-existing-model)
  - [Telemetry with .NET Aspire and Azure Application Insights](#telemetry-with-net-aspire-and-azure-application-insights)
- [Resources](#resources)
- [Video Recordings](#video-recordings)
- [Guidance](#guidance)
  - [Costs](#costs)
  - [Security Guidelines](#security-guidelines)
- [Resources](#resources)

[![overall explainer video](./images/90GenAiNetVideo.png)](https://aka.ms/genainnet/videos/lesson4-overview)

## Features

**GitHub CodeSpaces:** This project is designed to be opened in GitHub Codespaces, making deployment easy directly in the browser.

Perform **Keyword Search**:

![Keyword Search Demo](./images/05eShopLite-SearchKeyWord.gif)

Perform **Semantic Search**:

![Semantic Search Demo](./images/06eShopLite-SearchSemantic.gif)

Engage with **Real-time Audio Chat** for product inquiries:

![Realtime Audio Demo](./images/07eShopLite-RealtimeAudio.gif)

The Aspire Dashboard to check the running services:

![Aspire Dashboard to check the running services](./images/10AzureResources.png)

The Azure Resource Group with all the deployed services:

![Azure Resource Group with all the deployed services](./images/15AspireDashboard.png)

## Architecture Diagram

Updated architecture diagram showing all the components:

*Note: A new architecture diagram showing the RealtimeStore component must be created.*

## Getting Started

The solution is located in the `./src` folder. The main solution is **[eShopLite-Realtime-SemanticSearch](./src/eShopLite-Realtime-SemanticSearch)**.

## Deploying

Once you've opened the project in [Codespaces](#github-codespaces), or [locally](#run-locally), you can deploy it to Azure.

From a Terminal window, open the folder with the clone of this repo and run the following commands.

1. Login to Azure:

    ```shell
    azd auth login
    ```

1. Provision and deploy all the resources:

    ```shell
    azd up
    ```

    It will prompt you to provide an `azd` environment name (like "eShopLite"), select a subscription from your Azure account, and select a [location where OpenAI the models gpt-4o-mini, gpt-4o-realtime and ADA-002 are available](https://azure.microsoft.com/explore/global-infrastructure/products-by-region/?products=cognitive-services&regions=all) (like **"eastus2"** or **"swedencentral"**).

1. When `azd` has finished deploying, you'll see the list of resources created in Azure and a set of URIs in the command output.

1. Visit the **store** URI, and you should see the **eShop Lite app**! 🎉

1. Visit the **realtimestore** URI, and you should see the **eShop Lite Audio Chat App**! 🎉

1. This is an example of the command output:

![Deploy Azure Complete](./images/20AzdUpConsoleComplete.png)

***Note:** The deploy files are located in the `./src/eShopAppHost/infra/` folder. They are generated by the `Aspire AppHost` project.*

### GitHub CodeSpaces

- Create a new  Codespace using the `Code` button at the top of the repository.

- The Codespace creation process can take a couple of minutes.

- Once the Codespace is loaded, it should have all the necessary requirements to deploy the solution.

### Run Locally

To run the project locally, you'll need to make sure the following tools are installed:

- [.NET 9](https://dotnet.microsoft.com/downloads/)
- [Git](https://git-scm.com/downloads)
- [Azure Developer CLI (azd)](https://aka.ms/install-azd)
- [Visual Studio Code](https://code.visualstudio.com/Download) or [Visual Studio](https://visualstudio.microsoft.com/downloads/)
  - If using Visual Studio Code, install the [C# Dev Kit](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.csdevkit)
- .NET Aspire workload:
    Installed with the [Visual Studio installer](https://learn.microsoft.com/dotnet/aspire/fundamentals/setup-tooling?tabs=windows&pivots=visual-studio#install-net-aspire) or the [.NET CLI workload](https://learn.microsoft.com/dotnet/aspire/fundamentals/setup-tooling?tabs=windows&pivots=visual-studio#install-net-aspire).
- An OCI compliant container runtime, such as:
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) or [Podman](https://podman.io/).

### Run the solution

Follow these steps to run the project, locally or in CodeSpaces:

- Navigate to the Aspire Host folder project using the command:

  ```bash
  cd ./src/eShopAppHost/
  ```

- Set the user secrets for the Azure OpenAI connection string as explained in the [Local development using existing models](#local-development-using-an-existing-model) section.

- Run the project:

  ```bash
  dotnet run
  ````

### Local development using existing models

In order to use existing models: gpt-4o-mini, gpt-4o-realtime and text-embedding-ada-002, you need to define the specific connection string in the `Products` and `RealtimeStore` projects.

Add a user secret with the configuration:

```bash
cd src/Products

dotnet user-secrets set "ConnectionStrings:openai" "Endpoint=https://<endpoint>.openai.azure.com/;Key=<key>;"

cd ..
cd src/RealtimeStore

dotnet user-secrets set "ConnectionStrings:openai" "Endpoint=https://<endpoint>.openai.azure.com/;Key=<key>;"
```

This Azure OpenAI service must contain:

- a `gpt-4o-mini` model named **gpt-4o-mini**
- a `gpt-4o-realtime-preview` model named **gpt-4o-realtime-preview**
- a `text-embedding-ada-002` model named **text-embedding-ada-002**

To use these services, the `program.cs` file will load the information from the USer secrets instead of getting this information from the Aspire deploy.

```csharp
var azureOpenAiClientName = "openai";
builder.AddAzureOpenAIClient(azureOpenAiClientName);
```

### Telemetry with .NET Aspire and Azure Application Insights

The eShopLite solution leverages the Aspire Dashboard and Azure Application Insights to provide comprehensive telemetry and monitoring capabilities

The **.NET Aspire Dashboard** offers a centralized view of the application's performance, health, and usage metrics. It integrates seamlessly with the Azure OpenAI services, allowing developers to monitor the performance of the `gpt-4o-mini` and `text-embedding-ada-002` models. The dashboard provides real-time insights into the application's behavior, helping to identify and resolve issues quickly.

![Aspire Dashboard](./images/40AspireDashboard.png)

**Azure Application Insights** complements the Aspire Dashboard by offering deep diagnostic capabilities and advanced analytics. It collects detailed telemetry data, including request rates, response times, and failure rates, enabling developers to understand how the application is performing under different conditions. Application Insights also provides powerful querying and visualization tools, making it easier to analyze trends and detect anomalies. 

![Azure Application Insights](./images/45AppInsightsDashboard.png)

By combining the Aspire Dashboard with Azure Application Insights, the eShopLite solution ensures robust monitoring and diagnostics, enhancing the overall reliability and performance of the application.

## Guidance

### Costs

For **Azure OpenAI Services**, pricing varies per region and usage, so it isn't possible to predict exact costs for your usage.
The majority of the Azure resources used in this infrastructure are on usage-based pricing tiers.
However, Azure Container Registry has a fixed cost per registry per day.

You can try the [Azure pricing calculator](https://azure.com/e/2176802ea14941e4959eae8ad335aeb5) for the resources:

- Azure OpenAI Service: S0 tier, gpt-4o-mini and text-embedding-ada-002 models. Pricing is based on token count. [Pricing](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/)
- Azure Container App: Consumption tier with 0.5 CPU, 1GiB memory/storage. Pricing is based on resource allocation, and each month allows for a certain amount of free usage. [Pricing](https://azure.microsoft.com/pricing/details/container-apps/)
- Azure Container Registry: Basic tier. [Pricing](https://azure.microsoft.com/pricing/details/container-registry/)
- Log analytics: Pay-as-you-go tier. Costs based on data ingested. [Pricing](https://azure.microsoft.com/pricing/details/monitor/)
- Azure Application Insights pricing is based on a Pay-As-You-Go model. [Pricing](https://learn.microsoft.com/azure/azure-monitor/logs/cost-logs).

⚠️ To avoid unnecessary costs, remember to take down your app if it's no longer in use, either by deleting the resource group in the Portal or running `azd down`.

### Security Guidelines

Samples in this templates uses Azure OpenAI Services with ApiKey and [Managed Identity](https://learn.microsoft.com/entra/identity/managed-identities-azure-resources/overview) for authenticating to the Azure OpenAI service.

The Main Sample uses Managed Identity](https://learn.microsoft.com/entra/identity/managed-identities-azure-resources/overview) for authenticating to the Azure OpenAI service.

Additionally, we have added a [GitHub Action](https://github.com/microsoft/security-devops-action) that scans the infrastructure-as-code files and generates a report containing any detected issues. To ensure continued best practices in your own repository, we recommend that anyone creating solutions based on our templates ensure that the [Github secret scanning](https://docs.github.com/code-security/secret-scanning/about-secret-scanning) setting is enabled.

You may want to consider additional security measures, such as:

- Protecting the Azure Container Apps instance with a [firewall](https://learn.microsoft.com/azure/container-apps/waf-app-gateway) and/or [Virtual Network](https://learn.microsoft.com/azure/container-apps/networking?tabs=workload-profiles-env%2Cazure-cli).

## Want to know more?

For detailed technical documentation about the real-time audio capabilities and components of this scenario, including implementation details, architecture patterns, and configuration guides, see the comprehensive documentation:

**[📚 View Complete Technical Documentation](./docs/README.md)**

The documentation includes:
- Real-time audio integration with OpenAI GPT-4o  
- Aspire orchestration for multiple service types
- Blazor Server real-time communication
- Audio processing and conversation management
- Multi-model AI service configuration
- Product context integration patterns

## Resources

- [GPT-4o Realtime API Quickstart](https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-quickstart)

- [eShopLite Semantic Search Repository](https://aka.ms/netaieshoplitesemanticsearch)

- [eShopLite Semantic Search Repository using Azure AI Search](https://aka.ms/netaieshoplitesemanticsearchazureaisearch)

## Video Recordings

- [eShopLite - Real-time Audio Chat Overview](https://learn.microsoft.com/shows/generative-ai-for-beginners-dotnet/practical-samples-eshoplite-with-real-time-audio)