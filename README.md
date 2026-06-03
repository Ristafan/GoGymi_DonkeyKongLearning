# DKL-ML4BD

This repository contains a Jupyter Notebook focused on predicting student exam performance and interpreting model decisions using Explainable AI (XAI) and Counterfactual Explanations. 

The project analyzes student interactions with an online learning platform—including quiz scores, page views, media engagement, and time management—to determine which behaviors correlate most strongly with academic success.

---

## 📋 Research Questions

The notebook is structured around answering two core research questions:
1. **Prediction:** Can we predict final exam performance using students' quiz results, mock exam results, and behavior within online lessons?
2. **Explainability:** Which features have the greatest influence on our models' predictions of whether a student will pass or fail?

---

## ⚙️ Data Preprocessing Pipeline

The notebook constructs its target variables and feature matrices through a multi-step preprocessing workflow.

### Target Preprocessing
The target label (`passed_exam`) is extracted from mock exams hosted on the platform:
* **Session-Level Threshold:** For any given mock exam session, students answering **40% or more** of the questions correctly are classified as passing (1), while others are classified as failing (0).
* **Student-Level Aggregation:** Since students can take multiple mock exams, individual attempts are aggregated per user. A **majority voting** rule determines the final label. 
* **Tie-Breaking:** If a student has an equal number of passing and failing mock exam sessions, the classification outcome of their **most recent attempt** serves as the tie-breaker.

The pipeline splits students into two distinct educational tracks based on data tags:
* **Langzeit Gymnasium** (Long-term track)
* **Kurzzeit Gymnasium** (Short-term track)

### Feature Engineering
Features are structured around logical lesson blocks called `lektionen`. Each `lektionen` encapsulates several sub-topics (`themen`), videos, and localized quizzes. For each student, four normalized metrics (bounded between 0 and 1) are calculated per lesson block:

| Feature | Description | Aggregation Rule |
| :--- | :--- | :--- |
| **Quiz Score** | Normalized performance across quizzes within a lesson block. | Uses the student's **best attempt** if multiple attempts exist. |
| **Lesson Visited Rate** | The proportion of page views completed by the student. | Number of unique `themen` pages visited divided by total pages in the block. |
| **Time Score** | An indicator of lesson mastery based on active time spent. | Calculated by comparing individual page durations against the global student median. |
| **Media Completion Rate** | Engagement tracking for embedded video/audio components (e.g., Vimeo events). | Binary threshold per video (full credit given if a core milestone percentage is reached), averaged across the block. |

---

## 🛠️ Tech Stack & Key Libraries

The notebook utilizes standard Python data science components alongside deep learning and explainability frameworks:
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Modeling:** `scikit-learn` (Logistic Regression, preprocessing, and classification metrics)
* **Deep Learning Framework:** `torch` (PyTorch for neural network implementation)
* **Explainable AI (XAI):** `dice_ml` (Diverse Counterfactual Explanations for ML models)
* **Visualization:** `matplotlib`, `seaborn`

---

## 📁 Repository Structure & Expected Data

The execution of this notebook relies on a localized data layout mapped relative to the project root:

```text
├── data/
│   ├── math_results.csv        # mock exam results used for targets
│   ├── math_questions.csv      # exam question metadata and tags
│   ├── quiz_results.csv        # localized lesson quiz performance
│   ├── pageviews.csv           # reading and browsing telemetry
│   ├── course_ids.csv          # course mappings
│   └── events/                 
│       ├── 0.csv               # mouse and scroll interactions
│
