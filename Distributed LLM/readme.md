# Distributed LLM Inference System with FastAPI

This project demonstrates a **distributed LLM architecture** using two independent microservices that communicate via REST APIs. The system performs **LLM-based text generation, topic classification, and code extraction**.

The system uses two **FastAPI services** running on separate environments, connected through HTTP requests and exposed using **ngrok tunnels**.

---
## Architecture

Client
│
▼
Notebook 1 (LLM Generation Service)
│
▼
Notebook 2 (Topic Classification + Code Extraction Service)
---

## Workflow

1. User sends a prompt to **Service 1**

2. Service 1 generates text using **Llama 3.1 8B Instruct**

3. Generated text is sent to **Service 2**

4. Service 2:
   - Classifies topic using **facebook/bart-large-mnli**
   - Validates topic confidence
   - Extracts code snippets

5. Final response is returned to the client

## Example Client Request

### PowerShell

```powershell
$body = @{
    prompt = "Explain dynamic programming"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://your-service-url/generate/" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
