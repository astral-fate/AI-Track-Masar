 # 🤖 MASAR AI - Advanced NLP Course: From Transformers to RAG

## This 3-week intensive workshop provides a comprehensive, hands-on journey into modern Natural Language Processing (NLP) and AI.

The course begins with the foundational concepts of language modeling and Transformers, progresses into the architecture and fine-tuning of Large Language Models (LLMs), and culminates in building and deploying advanced Retrieval Augmented Generation (RAG) pipelines.

This curriculum is designed to bridge the gap between theory and practice, with each week's concepts reinforced by practical assignments and hands-on sessions.

## 📜 Table of Contents

* Course Overview
* Course Objectives
* Learning Outcomes
* Weekly Syllabus Breakdown
* Hands-On Assignments
* Key Technologies & Concepts

## 1. 📍 Course Overview

### Total Course Duration

| Category | Details |
| :--- | :--- |
| 🗓️ Frequency | 2 sessions per week (Sunday & Friday) |
| ⏰ Duration | 4 hours per session |
| 📅 Total Weeks | 3 |
| ⏱️ Total Hours | 24 hours (2 sessions/week * 3 weeks * 4 hours/session) |

## 2. 🎯 Course Objectives

The primary objectives of this course are to:

* **Understand Foundational Concepts:** Build a strong understanding of the evolution of NLP, from traditional N-gram models and RNNs to the revolutionary Transformer architecture and the "Attention" mechanism.
* **Master Modern Architectures:** Deconstruct and analyze modern LLM architectures (e.g., BERT, GPT, Llama), understanding their unique pre-training objectives (MLM vs. Causal LM) and architectural differences (RoPE, RMSNorm, GQA).
* **Analyze Computational Trade-offs:** Learn to calculate and plan for the significant memory (VRAM) requirements of training, fine-tuning, and inference (including the KV Cache).
* **Implement Efficient Fine-Tuning:** Gain practical skills in Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA and QLoRA, to adapt large models on consumer-grade hardware.
* **Build Advanced Search Systems:** Master the concepts behind semantic search, including contrastive learning, embedding models (Bi-Encoders vs. Cross-Encoders), and vector databases.
* **Develop and Deploy RAG:** Learn to architect and build a complete Retrieval Augmented Generation (RAG) pipeline from scratch to create chatbots that can answer questions using external, up-to-date knowledge, thereby reducing hallucination.
* **Apply Practical AI Skills:** Move beyond theory to gain hands-on experience in ASR, prompt engineering, ICL (zero/few-shot), and API-based model serving (vLLM, Groq).

## 3. 🎓 Learning Outcomes

Upon successful completion of this 3-week course, participants will be able to:

* **Architect & Differentiate:** Confidently explain the architectural differences, trade-offs, and use cases for RNNs, Transformers, BERT, GPT, and Llama.
* **Calculate & Plan:** Accurately estimate the VRAM required for both training and inference of LLMs, accounting for model parameters, precision, optimizer states, and the KV Cache.
* **Implement PEFT:** Apply LoRA and QLoRA to efficiently fine-tune billion-parameter models on custom datasets using limited hardware.
* **Build RAG Pipelines:** Design, build, and evaluate a complete, end-to-end RAG system, including a Bi-Encoder for retrieval, a Vector Store for indexing, and a Cross-Encoder for reranking.
* **Engineer Prompts:** Skillfully apply In-Context Learning (ICL) principles (zero-shot, one-shot, and few-shot) and prompt engineering to solve new tasks without fine-tuning.
* **Deploy & Serve:** Use serving engines like vLLM and interact with LLM APIs (like Groq) to integrate models into practical applications.
* **Fine-Tune for Specific Tasks:** Successfully fine-tune models for diverse tasks, including Automatic Speech Recognition (ASR) and Sentiment Analysis.

## 4. 📚 Weekly Syllabus Breakdown

### Week 1: Foundations of Language Modeling & Transformers

* **Core Concepts:** Introduction to Language Models, Traditional LMs (N-grams), Probability (Chain Rule), and Evaluation (Perplexity).
* **Sequential Models:** Recurrent Neural Networks (RNNs), Vanishing/Exploding Gradients, LSTMs, and Sequence-to-Sequence (Seq2Seq) models.
* **The Transformer:** The "Attention Is All You Need" concept, Self-Attention, Multi-Head Attention, Positional Encodings, and the full Encoder-Decoder architecture.
* **Pre-training:** BERT (Encoder-only, Masked LM) vs. GPT (Decoder-only, Causal LM).
* **ASR Deep Dive:** Traditional ASR vs. End-to-End, Feature Extraction (MFCCs), CTC Loss, and modern models (Wav2Vec 2.0, Whisper).

### Week 2: Large Language Models (LLMs) & Fine-Tuning

* **Modern Architectures:** A recap of Decoder-Only models and key architectural evolutions in models like Llama (RoPE, RMSNorm, SwiGLU, GQA).
* **Memory Analysis:** Calculating VRAM for training (Parameters, Gradients, Optimizer States) and inference (Parameters, KV Cache) at different precisions (FP32, FP16, 4-bit).
* **PEFT:** Parameter-Efficient Fine-Tuning, LoRA (Low-Rank Adaptation), and QLoRA (Quantized LoRA).
* **Sampling & Serving:** Understanding how LLMs generate text (Greedy, Top-K, Top-P), Softmax Temperature, and serving with vLLM.
* **Prompting:** In-Context Learning (ICL), Prompt Engineering, and the difference between Zero-Shot, One-Shot, and Few-Shot learning.

### Week 3: Retrieval Augmented Generation (RAG)

* **Vector Embeddings:** Similarity & Distance Metrics (Cosine Similarity, Dot Product, L1/L2 Distance).
* **Embedding Models:** How they are trained using Contrastive Learning (Triplet Loss, infoNCE).
* **Query-Document Scoring:** The critical difference between:
    * **Bi-Encoders (Fast Retrieval):** Embed query and document separately.
    * **Cross-Encoders (Accurate Reranking):** Embed query and document together.
* **Semantic Search:** The full pipeline, including document processing (chunking), indexing in a Vector Store, and retrieval.
* **RAG Pipelines:** The complete end-to-end architecture (Retrieve, Rerank, Augment, Generate) to reduce hallucination and provide up-to-date, verifiable answers.

## 5. 💻 From Theory to Practice: Hands-On Assignments

This course's practical assignments are designed to directly connect lecture concepts to real-world implementation.

### 🎙️ Assignment 1: Fine-Tuning Wav2Vec 2.0 for ASR

* **Objective:** Develop a working Automatic Speech Recognition (ASR) model for Arabic.
* **Connects to (Week 1):** ASR, Wav2Vec 2.0, CTC Loss, Fine-Tuning.
* **Enhanced Experience:**
    * Preprocessed a large, real-world audio dataset (Arabic Common Voice).
    * Gained practical experience fine-tuning a large, self-supervised model (Wav2Vec2-base).
    * Implemented the CTC Loss function and used greedy decoding to transcribe speech.

### 💬 Assignment 2: In-Context Learning (ICL) for Sentiment Analysis

* **Objective:** Build a sentiment classifier *without* any fine-tuning, using only ICL.
* **Connects to (Week 2):** ICL, Prompt Engineering, Llama, LLM APIs (Groq).
* **Enhanced Experience:**
    * Practiced the art of Prompt Engineering.
    * Directly compared the performance of *Zero-Shot* vs. *Few-Shot* prompting.
    * Learned to benchmark and evaluate prompt performance on a test set.
    * Gained experience using a high-speed, served LLM API (Groq) to run inference.

### 🔎 Week 3 Practical: Building a Full RAG Pipeline

* **Objective:** Create a fully functional semantic search engine and RAG pipeline.
* **Connects to (Week 3):** All concepts (Metrics, Bi/Cross-Encoders, Vector Stores, RAG).
* **Enhanced Experience:**
    * Built a practical query-document scoring system.
    * Implemented the *Retriever (Bi-Encoder)* for fast, scalable search.
    * Implemented the *Reranker (Cross-Encoder)* to improve the accuracy of search results.
    * Combined all components into an end-to-end RAG system that answers questions based on retrieved documents.

## 6. 🔑 Key Technologies & Concepts

| Core Models & Architectures | Fine-Tuning & Training | Semantic Search & RAG |
| :--- | :--- | :--- |
| 🤖 Transformers (BERT, GPT, Llama) | 🔧 Parameter-Efficient Fine-Tuning (PEFT) | 💡 Retrieval Augmented Generation (RAG) |
| 🗣️ Automatic Speech Recognition (ASR) | ✨ LoRA & QLoRA | 🔎 Semantic Search |
| 🌊 Wav2Vec 2.0 | 📉 CTC Loss | 🗄️ Vector Databases |
| 🎙️ Whisper | 🤝 Contrastive Learning | ⚖️ Bi-Encoders vs. Cross-Encoders |
| | 💡 In-Context Learning (ICL) | 🧠 KV Cache Memory Management |
| | ✍️ Prompt Engineering | |
