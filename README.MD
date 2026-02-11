## AI RAG Study Assistant

This project is a Retrieval-Augmented Generation (RAG) study assistant built as part of my preparation for the AWS AI Practitioner certification.
It focuses on how real AI systems are wired together rather than chatbot demos — emphasizing retrieval correctness, data typing, and cloud-ready architecture.

The app ingests markdown notes, stores embeddings in a vector database, retrieves relevant context at query time, and optionally generates grounded responses using Amazon Bedrock.

## What This Project Demonstrates

End-to-end RAG pipeline (ingest → embed → retrieve → generate)

Correct use of pgvector for similarity search

Clear separation of retrieval, prompt construction, and generation

Bedrock-ready design with local fallback for development

Practical debugging of real-world AI integration issues

## Current Status (MVP Progress)

✅ FastAPI backend running locally

✅ PostgreSQL + pgvector running in Docker

✅ Markdown ingestion with chunking and overlap

✅ Embeddings generated and stored as vector(1024)

✅ Vector similarity search verified directly in Postgres

✅ Retrieval logic corrected for pgvector casting

✅ /ask endpoint wired for RAG-style queries

⚠️ Bedrock generation scaffolded (model access/IAM pending)