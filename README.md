# Autonomous Self-Reflective Agentic AI System

An end-to-end, technically robust agentic AI system built for the **Ciklum AI Academy**. The system integrates multi-format data retrieval (RAG), autonomous multi-step reasoning (ReAct loop), tool execution, and self-reflection/critique into a unified architecture.

---

## 🏗️ Architecture & Component Design

The agent is designed around 5 foundational pillars:

1. **Data Preparation & Contextualization**: Pre-processes structured and unstructured technical knowledge (presentation slides, lecture transcripts, and repository files) with semantic boundary chunking and provenance metadata.
2. **RAG Pipeline Design**: Combines semantic embeddings with vector retrieval to ground agent reasoning in factual data.
3. **Reasoning & Reflection (ReAct Loop)**: Follows an autonomous **Thought ➔ Action ➔ Observation ➔ Reflection** cycle. The agent evaluates intermediate findings, identifies knowledge gaps, and refines its output.
4. **Tool-Calling Mechanisms**:
   - `rag_search`: Queries the vector knowledge base for domain context.
   - `inspect_file`: Explores repository structure and inspects code files.
   - `evaluate_response`: Performs heuristic and automated evaluation against accuracy, relevance, and clarity metrics.
   - `generate_social_post`: Autonomously drafts a professional LinkedIn summary of the agent's capabilities.
5. **Evaluation**: Built-in self-scoring evaluating outputs on a 1–5 scale across **Accuracy**, **Relevance**, and **Clarity**, enforcing automated self-correction before output delivery.

See `architecture.mmd` for the Mermaid visual architecture diagram.

---

## 🚀 Quickstart & Setup (Windows PowerShell)

### 1. Installation
Clone the repository, set up a virtual environment, and install dependencies:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Autonomous Agent Demo
Run the complete agentic pipeline:

```powershell
python agent.py
```

---

## 📊 Evaluation & Verification Results

Execution output from `demo_output.log`:
- **Hybrid Search Analysis**: Evaluated at 5/5 Accuracy, 5/5 Relevance, 5/5 Clarity (Overall: 5.0).
- **ColPali Architecture**: Evaluated at 5/5 Accuracy, 5/5 Relevance, 5/5 Clarity (Overall: 5.0).
- **Production RAG Do's**: Evaluated at 5/5 Accuracy, 5/5 Relevance, 5/5 Clarity (Overall: 5.0).

---