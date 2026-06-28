# NovaTech Solutions - Enterprise AI RAG Engine

A production-grade, modular Retrieval-Augmented Generation (RAG) conversational system engineered with Python, LangChain, and OpenAI. This system ingests unstructured enterprise Markdown knowledge baselines, transforms data fragments into dense vector representations, and handles contextual multi-turn dialog under strict isolation and reference constraints.

---

## 🏗️ Architecture Blueprint

The application architecture decouples structural parsing and search index tasks from conversational flow engines to guarantee horizontal scalability.

* **Presentation Layer (CLI):** Driven by `main.py`. Manages the terminal-based REPL loop, handles runtime interrupts gracefully, and routes operational outputs.
* **Domain Orchestration Layer:** Driven by `core.Chatbot`. Consumes active contexts, evaluates historical chat message sequences, and manages instructions boundaries using `gpt-4o`.
* **Retrieval-Augmented Generation (RAG) Core:** Driven by `core.RAGSystem`. Leverages `UnstructuredMarkdownLoader` and `RecursiveCharacterTextSplitter` to partition raw documents into granular token segments.
* **Vector Database Engine:** Uses LangChain's transient `InMemoryVectorStore` matched with `text-embedding-3-small` multidimensional arrays.

---

## 📂 File Directory Structure

```text
├── main.py              # Application runtime gateway and entry point
├── requirements.txt     # Locked project library dependencies
├── .env                 # Environment configuration credentials storage
├── README.md            # Enterprise configuration and setup guide
├── documents/           # Target ingestion directory for data storage
│   ├── documento1.md    # Private corporate context parameters
│   └── documento2.md    # Corporate policies and human resource procedures
└── core/                # Encapsulated application core packages
    ├── __init__.py      # Package constructor and public interfaces
    ├── rag_system.py    # Text parsing and vector processing engine
    └── chatbot.py       # Conversation execution and LLM control boundaries