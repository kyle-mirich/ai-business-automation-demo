# AI Research Papers for RAG Chatbot

This directory contains AI research papers for the RAG chatbot demo.

## Recommended Papers to Download

Download these foundational AI papers (all free on arXiv):

### 1. **Attention Is All You Need** (The Transformer Paper)
- **arXiv ID:** 1706.03762
- **Download:** https://arxiv.org/pdf/1706.03762.pdf
- **Save as:** `attention_is_all_you_need.pdf`
- **Why:** The foundational paper that introduced Transformers

### 2. **BERT: Pre-training of Deep Bidirectional Transformers**
- **arXiv ID:** 1810.04805
- **Download:** https://arxiv.org/pdf/1810.04805.pdf
- **Save as:** `bert.pdf`
- **Why:** Revolutionized NLP with bidirectional pre-training

### 3. **Language Models are Few-Shot Learners** (GPT-3)
- **arXiv ID:** 2005.14165
- **Download:** https://arxiv.org/pdf/2005.14165.pdf
- **Save as:** `gpt3_few_shot_learners.pdf`
- **Why:** Demonstrated the power of large language models

### 4. **An Image is Worth 16x16 Words: Transformers for Image Recognition** (ViT)
- **arXiv ID:** 2010.11929
- **Download:** https://arxiv.org/pdf/2010.11929.pdf
- **Save as:** `vision_transformer.pdf`
- **Why:** Extended Transformers to computer vision

### 5. **Chain-of-Thought Prompting Elicits Reasoning** (Optional)
- **arXiv ID:** 2201.11903
- **Download:** https://arxiv.org/pdf/2201.11903.pdf
- **Save as:** `chain_of_thought.pdf`
- **Why:** Modern prompting technique for better reasoning

## Quick Download Instructions

```bash
# Navigate to this directory
cd "data/papers"

# Download papers (using curl or wget)
curl -o attention_is_all_you_need.pdf https://arxiv.org/pdf/1706.03762.pdf
curl -o bert.pdf https://arxiv.org/pdf/1810.04805.pdf
curl -o gpt3_few_shot_learners.pdf https://arxiv.org/pdf/2005.14165.pdf
curl -o vision_transformer.pdf https://arxiv.org/pdf/2010.11929.pdf
```

## What the Chatbot Can Answer

Once papers are downloaded, the RAG chatbot can answer questions like:
- "Explain the transformer architecture"
- "What is self-attention?"
- "How does BERT differ from GPT?"
- "What are the key innovations in Vision Transformers?"
- "Compare multi-head attention mechanisms"

## Note

The ChromaDB vector database will be automatically created in `data/chroma_db/` when you first run the RAG chatbot demo.
