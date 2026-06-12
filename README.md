# AI-Supported Pre-Process Reflection — Thesis Prototype

Source-bound chatbot prototype developed as the artifact for the master's
thesis *AI-Supported Pre-Process Reflection for Renewable Energy
Infrastructure Projects: Designing and Evaluating a Source-Bound Chatbot*
(R. Amplo, HSLU, 2026).

Reference case: Windpark Lindenberg (Switzerland).

## Live Prototype

https://apparent-store-482769.framer.app

## What This Repo Contains

The FastAPI backend that powers the deployed chatbot. It implements a
RAG pipeline with BM25 retrieval over a curated knowledge base of Swiss
energy policy and project documents, and generates responses via the
OpenAI API (GPT-4o).

Architecture, design decisions, evaluation, and limitations are
documented in the thesis (Chapters 3–5).

## Local Setup

​```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uvicorn main:app --reload
​```

Backend runs at `http://localhost:8000`. The frontend is hosted separately
on Framer and is not part of this repository.

## Deployment

Auto-deployed to Railway from the `Development` branch.

## AI Use Disclosure

Generative AI (Claude) was used for coding support, as declared in the
thesis (p. 361). All output was reviewed and revised by the author.

## Contact

Raffaele Amplo — raffaele.amplo@stud.hslu.ch
