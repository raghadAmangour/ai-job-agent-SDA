# 🤖 AI Job Agent

An AI-powered job matching application that helps candidates discover relevant job opportunities by combining **LLM-based profile extraction, semantic search, rule-based filtering, deterministic matching, cross-encoder reranking, and an interactive AI agent**.

The project takes the original Phase 1–8 pipeline developed in Google Colab notebooks and turns it into a single **Streamlit web application** where a candidate can upload a resume, specify job preferences, receive ranked job matches, understand why each job was recommended, and ask questions about the results.

---

## 🎯 Project Idea

Many job-search systems rely heavily on keyword matching, which can miss semantically similar skills and experience described using different terminology.

For example, a candidate may have experience in:

> Machine Learning, Python, SQL, and Data Analysis

while a job description may describe similar requirements using different wording.

AI Job Agent combines multiple matching techniques to understand both the **candidate** and the **job description** more effectively.

The application answers questions such as:

* Which jobs are relevant to my background?
* Do I meet the requirements?
* Which skills do I already have?
* Which skills am I missing?
* Why was a specific job recommended?
* What should I learn to become a stronger candidate?
* Which of my matched jobs is more relevant?
* What jobs are available based on my preferences?

---

# 🔄 How It Works

The application follows a multi-stage pipeline:

```text
                    Candidate Resume
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Phase 2             │
                 │ Candidate Profile   │
                 │ + Embedding         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Phase 3             │
                 │ Semantic Retrieval  │
                 │ Embedding Search    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Phase 4             │
                 │ Hard Filtering      │
                 │ Preferences         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Phase 5             │
                 │ Match Scoring       │
                 │ + Skill Analysis    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Phase 6             │
                 │ Semantic Reranking  │
                 │ Cross-Encoder       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Phase 7             │
                 │ Job Explanations    │
                 │ + Evidence Checks   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Phase 8             │
                 │ AI Chat Agent       │
                 └─────────────────────┘
```

Each phase has a different responsibility.

---

# 🧩 Pipeline

## Phase 1 — Job Data Preparation

Phase 1 is responsible for preparing the job database.

It is **not executed inside the Streamlit application**.

The prepared data is stored in:

```text
data/
├── jobs_prepared.parquet
├── job_embeddings.npy
├── job_ids.npy
└── skill_vocabulary.parquet
```

The current dataset contains approximately **9,619 job postings**.

The application uses these precomputed files instead of rebuilding the entire job dataset every time a candidate runs an analysis.

To update the dataset, Phase 1 can be run separately and the generated files can then replace the files inside `data/`.

---

## Phase 2 — Candidate Profile

The candidate uploads a resume in:

* PDF
* DOCX
* TXT
* Markdown

The application extracts structured information from the resume using an LLM.

The resulting candidate profile contains information extracted from the resume, such as:

* Skills
* Education
* Experience
* Job titles
* Professional background
* Other relevant candidate attributes

The application also creates a **candidate embedding** representing the candidate's professional profile in vector space.

This embedding is later used for semantic job retrieval.

---

## Phase 3 — Semantic Retrieval

Instead of comparing the candidate against every job using simple keywords, the system uses embeddings to perform semantic retrieval.

The candidate embedding is compared with the precomputed job embeddings using **cosine similarity**.

Conceptually:

```text
Candidate Embedding
        │
        ▼
Compare with job embeddings
        │
        ▼
Similarity scores
        │
        ▼
Top-K candidate jobs
```

This stage produces a smaller set of potentially relevant jobs for the next stages.

The similarity calculation is performed locally using NumPy after the embeddings have been created.

---

## Phase 4 — Hard Filtering

Semantic similarity alone is not enough.

A job can be semantically relevant but still violate the candidate's actual preferences.

For example:

```text
Candidate:
Country: Saudi Arabia
City: Riyadh
Work Arrangement: Remote
Employment Type: Full-time
Salary: 8,000–12,000 SAR
```

The application applies explicit constraints such as:

* Country
* City
* Relocation preference
* Work arrangement
* Employment type
* Nationality
* Salary range

This stage removes jobs that do not satisfy the relevant constraints.

The filtering is intentionally separate from semantic similarity so that user-defined requirements are treated as explicit constraints rather than being left entirely to an AI model.

---

## Phase 5 — Candidate–Job Matching

The remaining jobs are evaluated against the candidate profile using a deterministic matching process.

The system evaluates factors such as:

* Skills
* Qualifications
* Education
* Other structured candidate/job attributes

The result is an explainable **Match Score from 0 to 100**.

The system also identifies relevant skill information, including:

```text
Matching Skills
Missing Skills
```

This provides a structured view of how the candidate compares with each job.

No embedding model or LLM is required for the core scoring calculation in this phase.

---

## Phase 6 — Semantic Reranking

An optional Cross-Encoder reranker is used to further evaluate and reorder the shortlisted jobs based on candidate–job relevance.

The application uses:

```text
Alibaba-NLP/gte-multilingual-reranker-base
```

The reranker receives the candidate and job information and evaluates their semantic relationship more directly.

The result is used to reorder the shortlisted jobs before generating the final explanations.

This stage is optional in the Streamlit interface:

> **Use semantic reranking**

Users can disable it if they need faster execution or if the deployment environment has limited resources.

---

## Phase 7 — Job Explanations

After the final jobs have been selected, the application generates a human-readable explanation for each result using an LLM.

The explanation is based on structured evidence already produced by the matching pipeline.

For example:

```text
Why this job matches

Matching Skills:
✓ Machine Learning
✓ Python

Missing Skills:
✗ Leadership
✗ Project Management

Development Suggestions:
- Strengthen leadership experience
- Build project management experience
```

The explanation stage includes validation to reduce unsupported claims and keep the explanation grounded in the available candidate/job evidence.

---

## Phase 8 — AI Chat Agent

The final stage provides an interactive AI assistant.

Instead of searching the entire job database again, the agent uses **precomputed results and local Python tools**.

The agent can answer questions such as:

```text
What are my top matching jobs?

Why was this job recommended?

Which skills am I missing?

Compare these two jobs.

What should I learn for this position?

Show me jobs in Riyadh.

What is my overall matching summary?
```

The agent uses function calling to access structured information such as:

* Top jobs
* Job explanations
* Skill gaps
* Learning plans
* Candidate summary
* Job comparisons
* Browseable job results

This keeps the conversational layer grounded in the results produced by the matching pipeline.

---

# ⚙️ User Preferences

Candidates can specify their preferences before running the analysis.

Available settings include:

| Preference          | Purpose                                 |
| ------------------- | --------------------------------------- |
| Desired countries   | Filter jobs by country                  |
| Desired cities      | Filter jobs by city                     |
| Willing to relocate | Relax location constraints              |
| Work arrangement    | Remote / Hybrid / On-site               |
| Employment type     | Full-time / Part-time / Contract / etc. |
| Nationality         | Match nationality requirements          |
| Minimum salary      | Define minimum salary expectation       |
| Maximum salary      | Define maximum salary expectation       |
| Salary currency     | Specify the salary currency             |
| Semantic reranking  | Enable/disable Cross-Encoder reranking  |
| Final jobs          | Number of jobs displayed                |

These preferences are primarily used during the filtering and ranking stages.

---

# 🏗️ Project Structure

```text
ai-job-agent-SDA/
│
├── app.py
│
├── src/
│   ├── config.py
│   ├── shared.py
│   ├── resume_reader.py
│   ├── phase2_profile.py
│   ├── phase3_retrieval.py
│   ├── phase4_filtering.py
│   ├── phase5_matching.py
│   ├── phase6_rerank.py
│   ├── phase7_explain.py
│   └── phase8_agent.py
│
├── data/
│   ├── jobs_prepared.parquet
│   ├── job_embeddings.npy
│   ├── job_ids.npy
│   └── skill_vocabulary.parquet
│
├── .streamlit/
│   └── secrets.toml.example
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies

The project combines several technologies:

### Application

* Streamlit
* Python

### AI / NLP

* OpenAI API
* LLM-based structured extraction
* Text embeddings
* Cross-Encoder reranking
* Function calling

### Data Processing

* Pandas
* NumPy
* Parquet
* SQLite / structured local data where applicable

### Machine Learning

* Cosine similarity
* Embedding retrieval
* Deterministic matching
* Cross-Encoder semantic reranking
* Skill-gap analysis

### Deployment

* GitHub
* Streamlit Community Cloud

---

# 📊 Current Job Dataset

The application currently uses a prepared dataset containing approximately:

**9,619 job postings**

The job database and embeddings are generated during the offline data preparation stage and are then consumed by the Streamlit application.

This allows the deployed application to focus on candidate matching instead of rebuilding the entire job corpus for every user.

---

# 💡 Design Principles

The project intentionally combines different approaches instead of relying on a single AI model.

### LLMs are used for:

* Candidate profile extraction
* Job explanations
* Conversational interaction

### Embeddings are used for:

* Semantic candidate–job retrieval

### Rule-based logic is used for:

* User preference filtering
* Salary constraints
* Location constraints
* Employment constraints
* Deterministic matching

### Cross-Encoder is used for:

* Semantic reranking of shortlisted jobs

This separation makes the pipeline more structured, explainable, and easier to evaluate.

---

# 🎯 Project Goal

The goal of AI Job Agent is to provide a more structured job-search experience by combining:

**Semantic understanding + explicit user preferences + explainable matching + conversational AI**

Instead of simply returning jobs that contain similar keywords, the system creates a multi-stage matching process that considers both the candidate's background and their stated preferences.


---

## 👩‍💻 Project

**AI Job Agent — SDA / WeCloudData**

Built as a multi-phase AI and data project integrating information retrieval, NLP, machine learning, deterministic scoring, and conversational AI into a single deployed application.
