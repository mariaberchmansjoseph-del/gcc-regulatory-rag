
---
title: GCC Regulatory RAG
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: true
---

# GCC Regulatory Intelligence RAG
AI-powered Q&A over UAE financial regulations.

## Evaluation
- Faithfulness: 0.926
- Answer Relevancy: 0.961

# GCC Regulatory Intelligence RAG

AI-powered Q&A over UAE financial regulations using hybrid retrieval.

## Documents
- CBUAE Financial Stability Report 2024 & 2025
- IMF UAE Article IV 2024
- DFSA AML Rulebook 2024
- SCA Amendments 2024

## Architecture
- **Retrieval**: FAISS dense + BM25 sparse + Reciprocal Rank Fusion
- **Graph**: NetworkX knowledge graph (208 nodes, 967 edges)
- **Generation**: GPT-4o-mini with citation enforcement
- **Evaluation**: Faithfulness 0.926 · Answer Relevancy 0.961

## API
- `POST /query` — ask a regulatory question
- `GET /health` — service status
- `GET /docs` — Swagger UI
