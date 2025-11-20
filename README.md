# 🚀 End-to-End Machine Learning Engineering Portfolio

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-GenAI_Framework-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)

## 👨‍💻 About Me
**Hi, I'm Aakash.** A prolific learner
I am a Data scientist focused on building real, working ML systems. Currently Exploring ML Engineer and AI solutions. Professionally I’ve designed and deployed credit-risk and decision models end-to-end—data prep, feature engineering, training, optimization, explainability, monitoring, and API deployment. My work includes scorecards, acquisition and collection models, RCA-driven improvements, and robust monitoring using CSI/PSI and variable-quality analytics.

I’m building a public repository of production-grade projects: clean pipelines, MLflow-powered experiment tracking, AutoML explorations, and fully packaged services ready to integrate into real products. I enjoy turning ideas into reproducible, scalable implementations and exploring the boundaries of applied AI—risk analytics, RL, financial decision systems, forecasting, and XAI.

Always learning. Always building. This space will keep growing with new models, tools, and experiments that reflect my journey in applied machine learning.

This repository serves as a **Monorepo of Microservices**, demonstrating my expertise across the full AI lifecycle:
1.  **Supervised Learning** (Financial Fraud)
2.  **Vector Search & Recommenders** (Hybrid Engines)
3.  **Generative AI & RAG** (Legal/Banking Compliance)
4.  **MLOps & Governance** (Automated Pipelines)

---

## 🌟 Flagship Project: Sentinel Credit Risk Engine
**Folder:** [`/sentinel-credit-risk`](./sentinel-credit-risk/)

A "Lead Engineer" level MLOps platform designed for the banking sector. This is not just a model; it is a self-correcting system.

* **The Problem:** Credit risk models decay over time ("Drift") and require rigorous audit trails.
* **The Solution:** An automated pipeline that links Data Engineering, AutoML, and Governance.
* **Key Engineering:**
    * **DataOps:** Implemented **DVC** (Data Version Control) to track dataset versions like Git commits.
    * **Feature Engineering:** Built an industry-standard **Weight of Evidence (WoE)** & **Information Value (IV)** transformation pipeline.
    * **AutoML:** Integrated **Optuna** to automatically hunt for the best Hyperparameters.
    * **Tracking:** Used **MLflow** to log every experiment, metric, and artifact (model files).
    * **Governance:** Automated **Evidently AI** reports to detect data drift and generated **MkDocs** sites for model documentation.

---

## 💼 Specialized Microservices

### 1. Banking Compliance RAG Assistant
**Folder:** [`/banking-rag`](./banking-rag/)
* **Domain:** Generative AI / Fintech
* **Tech:** LangChain, OpenAI, ChromaDB, PyPDF
* **Description:** A Q&A system for complex loan policy PDF documents.
* **The "Wow" Factor:** Implemented strict **Guardrails**. The bot refuses to answer questions not found in the text and **must cite the exact page number** `[Page 12]` for every claim. This solves the "Hallucination" problem in high-stakes finance.

### 2. Hybrid Recommendation Engine
**Folder:** [`/recommendation-system`](./recommendation-system/)
* **Domain:** Personalization / Entertainment
* **Tech:** FAISS (Vector Search), SVD (Collaborative Filtering), Sentence-Transformers
* **Description:** A "Netflix-style" engine using the MovieLens dataset.
* **The "Wow" Factor:** Implemented a **Weighted Hybrid Architecture**. It combines **Content-Based Filtering** (Vector embeddings of plots) with **Collaborative Filtering** (SVD user-ratings) to solve the "Cold Start" problem while maintaining high accuracy.

### 3. Real-Time Fraud Detection
**Folder:** [`/fraud-detection`](./fraud-detection/)
* **Domain:** CyberSecurity / Payments
* **Tech:** XGBoost, FastAPI, Docker
* **Description:** A low-latency API that scores transactions in milliseconds.
* **The "Wow" Factor:** Specifically engineered to handle **Imbalanced Data** (using scale_pos_weight and probability calibration) and deployed as a containerized microservice suitable for Kubernetes.

---

## 🛠️ Technical Skills Matrix

| Area | Competencies | Tools Used |
| :--- | :--- | :--- |
| **MLOps & Engineering** | CI/CD, Containerization, Experiment Tracking, Data Versioning | Docker, GitHub Actions, MLflow, DVC, Poetry |
| **Generative AI** | RAG, Vector Databases, Prompt Engineering, Hallucination Control | LangChain, ChromaDB, OpenAI API, TikToken |
| **Data Science** | Feature Engineering (WoE/IV), AutoML, Drift Detection | Pandas, Optuna, Evidently AI, Scikit-Learn |
| **Backend / API** | REST APIs, Type Validation, Async Processing | FastAPI, Uvicorn, Pydantic |
| **Cloud & Deploy** | Serverless deployment, Microservices | Render, Shell Scripting |

---

## ⚡ Quick Start Guide

This repository uses a **Monorepo** structure. Each project is isolated with its own dependencies.

### Prerequisites
* Python 3.11+
* Poetry (`pip install poetry`)
* Docker (Optional)

### Setup
Each project is self-contained. To run a specific microservice, navigate to its folder (e.g., `cd sentinel-credit-risk`) and follow the local `README.md` instructions to install dependencies via Poetry and launch the API.

---


## 📫 Contact

I am currently open to **Senior ML Engineer** and **Data Scientist** roles.

* **LinkedIn:** [Link to your Profile]
* **Email:** [you@email.com]
* **Portfolio:** [Link to your Website]

---
*Built with ❤️, ☕, and Python.*