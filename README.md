# AI Job Agent

An AI-powered job matching assistant that analyzes a candidate's resume, retrieves relevant job opportunities, applies hard eligibility filters, ranks the remaining jobs, explains the match, and provides an interactive AI career assistant.

The project combines semantic retrieval, rule-based filtering, deterministic candidate-job matching, cross-encoder reranking, grounded job explanations, and an interactive AI agent.

---

## Project Idea

Finding suitable job opportunities can be time-consuming because a candidate needs to consider many factors at the same time, including:

* Skills
* Experience
* Education
* Qualifications
* Location
* Work arrangement
* Employment type
* Nationality
* Salary expectations

AI Job Agent automates this process through a multi-phase pipeline.

The system does not rely only on semantic similarity. It combines **semantic retrieval, explicit eligibility filtering, structured matching, reranking, and grounded explanations** to produce more useful and transparent job recommendations.

---

## How It Works

The application follows this pipeline:

```text
Candidate Resume
       │
       ▼
Phase 2 — Candidate Profile
       │
       ▼
Phase 3 — Embedding Retrieval
       │
       ▼
Phase 4 — Hard Filtering
       │
       ▼
Phase 5 — Candidate Matching
       │
       ▼
Phase 6 — Cross-Encoder Reranking
       │
       ▼
Phase 7 — Job Explanations
       │
       ▼
Phase 8 — AI Agent
       │
       ▼
Interactive Job Search & Career Assistance
```

**Phase 1** prepares the job dataset and precomputes job embeddings before the application is used.

**Phase 9** evaluates the behavior of the Phase 8 agent and is used for project evaluation rather than the production application pipeline.

---

# Pipeline

## Phase 1 — Job Data Preparation

Phase 1 prepares the raw job dataset for the rest of the pipeline.

The process includes:

* Cleaning and normalizing job data
* Structured extraction of job attributes
* Normalizing skills
* Preparing job descriptions
* Creating the skill vocabulary
* Generating job embeddings
* Saving reusable outputs for later phases

### Current Dataset

The current prepared dataset contains:

* **9,619 jobs**
* **57 columns**
* **1,536-dimensional job embeddings**
* **35,890 normalized skills**
* **100% successful structured extraction**

### Phase 1 Outputs

The application uses the following prepared files:

```text
data/
├── jobs_prepared.parquet
├── job_embeddings.npy
├── job_ids.npy
└── skill_vocabulary.parquet
```

These files allow the application to use the prepared job data without repeating the full preprocessing and embedding process every time the application starts.

---

## Phase 2 — Candidate Profile

Phase 2 processes the candidate's resume and converts it into a structured candidate profile.

### Supported Resume Formats

* PDF
* DOCX
* TXT
* Markdown

The system extracts information such as:

* Most recent title
* Previous titles
* Hard skills
* Soft skills
* Tools and technologies
* Languages
* Qualifications
* Experience level
* Years of experience
* Education level
* Education field
* Industries
* Candidate preferences

The extracted information is represented using structured Pydantic schemas.

The candidate profile is also converted into an embedding for semantic job retrieval.

### Candidate Embedding

The candidate embedding uses:

```text
text-embedding-3-small
```

with a dimensionality of:

```text
1536
```

---

## Phase 3 — Embedding Retrieval

Phase 3 performs semantic retrieval between the candidate and the prepared job dataset.

The candidate embedding is compared against the precomputed job embeddings using cosine similarity.

The system retrieves the highest-ranked candidates from the full job dataset before applying more detailed eligibility rules.

### Current Configuration

```text
Retrieval Top K: 150
```

This stage is designed to reduce the search space while preserving semantically relevant opportunities for the following phases.

---

## Phase 4 — Hard Filtering

Phase 4 applies explicit candidate preferences and job constraints to the retrieved jobs.

The filtering process considers:

* Country
* City
* Willingness to relocate
* Work arrangement
* Employment type
* Education level
* Nationality
* Salary expectations

Examples of filtering behavior include:

* If the candidate is willing to relocate, country and city restrictions can be skipped.
* Work arrangement is enforced when the job's work arrangement has been reliably identified.
* Education levels follow a defined hierarchy.
* Nationality requirements are handled explicitly.
* Salary filtering considers salary overlap and avoids rejecting jobs when salary or currency information is unavailable.

The goal of this phase is to remove jobs that do not satisfy important explicit constraints before detailed candidate-job scoring.

---

## Phase 5 — Candidate Matching

Phase 5 performs structured candidate-job matching on the jobs that survive the hard filters.

The matching process evaluates multiple criteria:

* Core skills
* Languages
* Experience fit
* Education field
* Qualifications

The system produces a structured matching result including:

* Match score
* Objective ranking score
* Scoring confidence
* Matched skills
* Missing skills
* Matched qualifications
* Missing qualifications
* Education matching information

The match score is a **requirement-coverage score**, not a probability of getting hired.

---

## Phase 6 — Cross-Encoder Reranking

Phase 6 performs a second ranking stage using a cross-encoder reranker.

### Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model was selected because the resume and job-description text used by the application is English and the model provides a lightweight reranking option suitable for the Streamlit deployment environment.

### Runtime Configuration

The project uses:

```text
Transformers: 4.53.3
```

The Transformers version is intentionally pinned to the tested Phase 6 runtime.

### Configuration

```text
Maximum pair tokens: 512
Final top jobs: 20
Initial batch size: 8
```

The reranker evaluates candidate-job text pairs and produces a refined ranking of the jobs that survived the earlier stages.

---

## Phase 7 — Job Explanations

Phase 7 generates grounded explanations for the final matched jobs.

For each job, the system can provide:

* Why the job matches the candidate
* Matched skills
* Missing skills
* Matched qualifications
* Missing qualifications
* Matched education
* Missing education
* Suggested learning areas

The explanations are generated from structured matching information rather than asking the language model to independently determine whether a job is suitable.

This helps keep the explanations tied to the actual matching results.

---

## Phase 8 — AI Agent

Phase 8 provides an interactive conversational agent on top of the previously computed results.

The agent can help the candidate with tasks such as:

* Viewing top matched jobs
* Understanding why a job matches
* Identifying missing skills
* Creating an improvement plan
* Preparing interview questions
* Comparing job information
* Answering questions about the available results

The agent uses local Python tools to access the previously generated results.

### Grounding Rules

The agent is instructed to:

* Retrieve job information before making factual claims about a specific job.
* Never invent a `job_id`.
* Never invent skills, qualifications, or education requirements.
* Explain limitations when requested information is unavailable.
* Avoid presenting a match score as a hiring probability.
* Avoid making unsupported guarantees.
* Avoid claiming that candidate preferences were changed when they require rerunning earlier phases.
* Respond in Arabic or English according to the candidate's language.
* Base job explanations on the Phase 7 output.
* Present differences between jobs factually rather than making an overall judgment about which job is better.

The agent uses:

```text
LangGraph
LangChain
OpenAI API
Pydantic
```

---

# Phase 9 — Evaluation

Phase 9 evaluates whether the Phase 8 agent behaves according to its grounding and system-prompt rules.

It is an **evaluation phase**, not part of the production Streamlit pipeline.

### Evaluation Goals

The evaluation checks whether the agent:

* Selects the appropriate tool for different user requests.
* Does not invent jobs or job IDs.
* Does not introduce unsupported hiring probabilities or guarantees.
* Handles invalid job IDs correctly.
* Handles requests outside the available capabilities gracefully.
* Handles ambiguous requests.
* Handles requests when no job data is loaded.

### Automated Checks

The evaluation automatically checks:

1. **Tool Selection**

   Whether the expected tool was called for each scripted prompt.

2. **Job ID Grounding**

   Whether job IDs mentioned in the response correspond to known jobs in the loaded Phase 7 data.

3. **Forbidden Phrasing**

   Whether the response contains prohibited patterns such as unsupported hiring percentages or guarantee language.

### Manual Review

Some aspects require human review, particularly:

* Whether the response is well written.
* Whether the response is actually helpful.
* Whether the explanation is grounded and understandable.
* Whether the response handles the user's request naturally.

### Scripted Test Cases

The evaluation covers cases including:

* Listing matched jobs
* Explaining a specific job
* Generating an improvement plan
* Generating interview preparation questions
* Handling an invalid job ID
* Handling an out-of-scope request
* Handling a request for a hiring percentage
* Handling an ambiguous request
* Handling an empty session with no loaded job data

### Phase 9 Outputs

The evaluation generates:

```text
evaluation_results.csv
evaluation_report.json
data_dictionary.txt
```

These outputs are used as supporting material for the project's final evaluation and report.

---

# User Preferences

Before running the matching process, candidates can customize their job preferences.

The available preferences include:

| Preference          | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| Country             | Preferred job country                                                   |
| City                | Preferred job city                                                      |
| Work Arrangement    | On-site, Remote, Hybrid, or Not Specified                               |
| Employment Type     | Full-time, Part-time, Contract, Internship, Temporary, or Not Specified |
| Willing to Relocate | Whether the candidate is open to relocation                             |
| Nationality         | Candidate nationality when relevant to job requirements                 |
| Expected Salary     | Minimum and maximum expected salary                                     |
| Salary Currency     | Expected salary currency                                                |
| Salary Period       | Hour, day, month, or year                                               |
| Availability        | Candidate availability                                                  |
| Notes               | Additional preferences                                                  |

Semantic reranking can also be enabled to refine the final ranking.

---

# Project Structure

The repository keeps the working project structure used by the application:

```text
ai-job-agent-SDA/
│
├── data/
│   ├── jobs_prepared.parquet
│   ├── job_embeddings.npy
│   ├── job_ids.npy
│   └── skill_vocabulary.parquet
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
├── notebooks/
│   ├── Phase1_Job_Data_Preparation_Final.ipynb
│   ├── Phase2_Candidate_Profile.ipynb
│   ├── Phase3_Embedding_Retrieval.ipynb
│   ├── Phase4_Hard_Filtering.ipynb
│   ├── Phase5_Candidate_Matching.ipynb
│   ├── Phase6_CrossEncoder_Reranking.ipynb
│   ├── Phase7.ipynb
│   ├── Phase8_Agent.ipynb
│   └── Phase9.ipynb
│
├── assets/
│   └── logo.png
│
├── .streamlit/
│   └── secrets.toml.example
│
├── app.py
├── requirements.txt
└── README.md
```

### Repository Components

* `app.py` — Streamlit application entry point.
* `src/` — Python modules used by the application pipeline.
* `data/` — Prepared job data and precomputed embeddings required by the application.
* `notebooks/` — Development and evaluation notebooks for the individual project phases.
* `assets/` — Application visual assets, including the project logo.
* `.streamlit/secrets.toml.example` — Example configuration for required secrets.
* `requirements.txt` — Python dependencies and tested runtime versions.
* `README.md` — Project documentation.

The notebooks are included for development, review, and reproducibility. They are not required for the normal Streamlit application runtime.

---

# Technologies

The project uses:

### Application

* Streamlit
* Python

### Data Processing

* Pandas
* NumPy
* PyArrow

### Resume Processing

* PDFPlumber
* Python-DOCX

### LLM and AI

* OpenAI API
* LangChain
* LangGraph
* Pydantic

### Retrieval and Embeddings

* OpenAI `text-embedding-3-small`
* NumPy-based cosine similarity

### Reranking

* Hugging Face Transformers
* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* PyTorch
* Accelerate
* Safetensors

---

# Installation and Setup

## 1. Install Dependencies

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The project uses the following main dependencies:

```text
streamlit>=1.38
pandas>=2.0
numpy>=1.26,<2.0
pyarrow>=15.0
openai>=1.40
pydantic>=2.6
pdfplumber>=0.11
python-docx>=1.1

transformers==4.53.3
accelerate>=1.0,<2
safetensors>=0.4
torch>=2.2,<3
```

The Transformers version is intentionally pinned to the tested Phase 6 runtime.

---

## 2. Configure the OpenAI API Key

Create the Streamlit secrets file:

```text
.streamlit/secrets.toml
```

based on:

```text
.streamlit/secrets.toml.example
```

The example file contains:

```toml
OPENAI_API_KEY = "sk-..."
```

Replace the placeholder with a valid OpenAI API key.

**Do not commit the real `secrets.toml` file or expose the API key publicly.**

---

## 3. Run the Application

From the project root:

```bash
streamlit run app.py
```

The application will start locally and provide the Streamlit interface.

---

# Application Data

The Streamlit application expects the prepared data files inside the `data/` directory:

```text
data/
├── jobs_prepared.parquet
├── job_embeddings.npy
├── job_ids.npy
└── skill_vocabulary.parquet
```

The application uses these files to load the prepared job dataset and precomputed embeddings.

The data paths are configured in:

```text
src/config.py
```

The current data directory is:

```python
DATA_DIR = "data"
```

---

# API Key and Privacy

The project requires an OpenAI API key for LLM-powered functionality.

The API key should be stored locally in Streamlit secrets rather than inside source code.

The repository should contain only:

```text
.streamlit/secrets.toml.example
```

and should **not** contain the real:

```text
.streamlit/secrets.toml
```

or any other file containing a private API key.

---

# Dataset Source

The job dataset used in the project was obtained from **Open Jobs by Elliott Dehn**.

**Source:** GitHub — Open Jobs by Elliott Dehn

The original dataset was processed, cleaned, and transformed during Phase 1 before being used by the application.

The prepared dataset used by the project contains 9,619 jobs and 57 columns.

The original dataset source should be acknowledged when redistributing or presenting the project.

---

# Design Principles

The project was designed around the following principles:

### 1. Structured Processing

Important candidate and job information is represented using structured schemas rather than relying entirely on free-form text.

### 2. Multi-Stage Matching

The system combines multiple stages:

```text
Semantic Retrieval
        ↓
Hard Eligibility Filtering
        ↓
Structured Matching
        ↓
Cross-Encoder Reranking
        ↓
Grounded Explanation
        ↓
Conversational Assistance
```

### 3. Grounded AI Responses

The conversational agent is instructed to rely on previously computed results and available tools rather than inventing information.

### 4. Separation of Responsibilities

Each phase has a specific responsibility:

* Data preparation
* Candidate extraction
* Retrieval
* Filtering
* Matching
* Reranking
* Explanation
* Agent interaction
* Evaluation

### 5. Reusable Precomputed Data

Job embeddings and prepared job data are generated before runtime so that the application does not need to repeat the entire dataset preparation process for every user session.

### 6. Evaluation and Validation

The final agent is evaluated using scripted test cases and automated grounding checks, with manual review for response quality.

---

# Current Job Dataset

The current prepared dataset used by the project contains:

| Item                          |  Value |
| ----------------------------- | -----: |
| Jobs                          |  9,619 |
| Columns                       |     57 |
| Job embedding dimensions      |  1,536 |
| Skill vocabulary              | 35,890 |
| Structured extraction success |   100% |

These values describe the current prepared dataset used during the project's development and evaluation.

---

# Development Notebooks

The `notebooks/` directory contains the development notebooks used to build and evaluate the project phases:

```text
Phase1_Job_Data_Preparation_Final.ipynb
Phase2_Candidate_Profile.ipynb
Phase3_Embedding_Retrieval.ipynb
Phase4_Hard_Filtering.ipynb
Phase5_Candidate_Matching.ipynb
Phase6_CrossEncoder_Reranking.ipynb
Phase7.ipynb
Phase8_Agent.ipynb
Phase9.ipynb
```

The notebooks document the development process and can be used for review.

The production Streamlit application uses the Python modules under `src/` rather than requiring the notebooks to run.

---

# Project Goal

The goal of AI Job Agent is to provide a structured and transparent way for candidates to explore job opportunities based on their resume, skills, experience, education, and preferences.

Instead of relying on a single similarity score, the system combines retrieval, explicit filtering, structured matching, reranking, grounded explanations, and an interactive agent to help candidates understand the available opportunities and identify areas for improvement.

---

# Final Project Components

The final project consists of:

```text
AI Job Agent
│
├── Streamlit Application
├── Prepared Job Dataset
├── Precomputed Job Embeddings
├── Matching Pipeline
├── Cross-Encoder Reranker
├── Grounded Job Explanations
├── Interactive AI Agent
├── Evaluation Pipeline
└── Development Notebooks
```

---

# Attribution

Developed as part of the SDA / WeCloudData project.

The job dataset used in the project is based on Open Jobs by Elliott Dehn.
