# Complaint Auto-Routing System

An AI/ML-powered government complaint management system that automatically routes citizen complaints to the most suitable officer while predicting priority, estimating resolution time, and finding similar past complaints.

Built entirely with local, offline models — no external APIs.

---

## System Architecture
Input (Text / Audio / Video)
↓
Language Detection + Translation (langdetect + deep-translator)
↓
Multilingual Embeddings (paraphrase-multilingual-mpnet-base-v2)
↓
┌──────────────────────────────────────┐
│         Multi-Task ML Pipeline        │
├──────────────┬───────────────────────┤
│ Priority     │ ETA          │ Officer │
│ Classifier   │ Regressor    │ Router  │
│ (RF, F1=0.87)│ (MAE=1.29d)  │ (FAISS) │
└──────────────┴───────────────────────┘
↓
Similarity Search (FAISS Vector Search)
↓
Unified JSON Response via FastAPI

---

## Features

- **Multilingual Support** — Handles English, Hindi, Malayalam, Marathi, Tamil, and more. Auto-detects language and translates to English before processing.
- **Audio/Video Input** — Transcribes speech using OpenAI Whisper (runs fully offline) and extracts audio from video using ffmpeg.
- **Officer Routing** — Weighted scoring: 60% semantic similarity + 30% officer performance score + 10% workload availability. Overloaded officers automatically excluded.
- **Priority Prediction** — Random Forest Classifier trained on 3,300 complaints. 95.45% test accuracy, F1 score 0.95 on unseen data.
- **ETA Estimation** — Random Forest Regressor combining text embeddings + priority + category. MAE of 0.85 days.
- **Similarity Search** — FAISS vector index across all 3,300 past complaints. Returns top 5 semantically similar cases.
- **Real-time API** — FastAPI with all models preloaded at startup. Average response time ~250ms.

---

## Architecture Decisions

**Why paraphrase-multilingual-mpnet-base-v2 for embeddings?**
Trained on 50+ languages including Indian languages. Outperformed English-only models (cosine similarity improved from 0.47 to 0.69 on Indian language complaint pairs). Runs fully offline.

**Why Random Forest over deep learning for classification?**
With 3,300 samples, Random Forest with 5-fold cross validation gave consistent 95% CV accuracy without overfitting. It's robust to noisy labels and significantly faster at inference than deep neural networks.

**Why FAISS over a traditional database for similarity search?**
Traditional databases match exact characters. FAISS performs cosine similarity search on 768-dimensional vectors, enabling semantic matching — "water shortage" correctly matches "no water supply" even though no words overlap.

**Why preload all models at startup?**
Initial implementation loaded models per request, causing 24 second response times. Preloading all models into memory at startup reduced response time to ~250ms — a 99% improvement.

**Why add noise to training data?**
Initial synthetic data produced 1.0 training accuracy — a clear sign of overfitting. Adding 5% random priority noise produced realistic 95% test accuracy with proper generalization.

---

## Project Structure
```text
complaint-routing-system/
├── api/                    # FastAPI application
│   ├── main.py             # All endpoints + model preloading
│   ├── schemas.py          # Pydantic request/response schemas
│   └── static/index.html   # Web interface
├── data/
│   ├── generate_data.py    # Synthetic data generation
│   ├── officers.json       # 20 officers across 6 departments
│   └── complaints.json     # 500 training complaints
├── pipeline/
│   ├── translate.py        # Language detection + translation
│   ├── ingest.py           # Text/audio/video input handling
│   └── features.py         # Multilingual embeddings
├── models/
│   ├── priority_classifier.py  # RF classifier (high/medium/low)
│   ├── eta_regressor.py        # RF regressor (days)
│   ├── officer_router.py       # Weighted semantic officer matching
│   └── similarity_search.py    # FAISS vector search
├── saved_models/           # Trained model artifacts
├── evaluation/
│   ├── metrics.py          # Evaluation framework
│   └── reports/            # Generated charts and reports
└── tests/                  # Unit tests
```

---

## Setup and Installation

**Requirements:** Python 3.11, Git

```bash
# Clone the repository
git clone https://github.com/soooooooraj/complaint-auto-routing-system.git
cd complaint-auto-routing-system

# Create virtual environment
py -3.11 -m venv venv311
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Running the System
Step 1 — Generate data and train models:
```bash
python data/generate_data.py
python models/priority_classifier.py
python models/eta_regressor.py
python models/officer_router.py
python models/similarity_search.py
```

Step 2 — Start the API server:
```bash
uvicorn api.main:app --reload --port 8000
```

Step 3 — Open the web interface:
http://localhost:8000

API Endpoints
| Method | Endpoint | Description |
| --- | --- | --- |
| GET | /health | System health check |
| GET | /stats | Complaint statistics |
| GET | /officers | All officers and workload |
| POST | /complaint/text | Route a text complaint |
| POST | /complaint/audio | Route an audio complaint |
| POST | /complaint/video | Route a video complaint |

Example request:
```bash
curl -X POST http://localhost:8000/complaint/text \
  -H "Content-Type: application/json" \
  -d '{"text": "No water supply for 3 days in our colony"}'
```

Example response:
```json
{
  "complaint_text": "No water supply for 3 days in our colony",
  "detected_language": "en",
  "priority": "high",
  "priority_confidence": 0.46,
  "eta_days": 2,
  "assigned_officers": [
    {
      "officer_id": "OFC011",
      "name": "Brijesh Varty",
      "department": "Water Supply",
      "final_score": 0.48,
      "current_workload": 1
    }
  ],
  "similar_complaints": [...],
  "processing_time_ms": 450.21
}
```

Evaluation Results
| Metric | Value |
| --- | --- |
| Priority Classification Accuracy | 95.45% |
| Priority F1 Score (weighted) | 0.95 |
| ETA Mean Absolute Error | 0.85 days |
| ETA RMSE | 1.12 days |
| Average API Response Time | ~250ms |
| Training Complaints | 2,640 |
| Test Complaints | 660 |
| Cross-Validation Mean | 95.19% |

Stress Test
The model was tested on 10 deliberately ambiguous complaints with no obvious keywords. Confidence dropped from ~0.60 on standard complaints to 0.35-0.47 on ambiguous ones — demonstrating the model correctly identifies uncertainty. In production, a confidence threshold of 0.50 would flag low-confidence predictions for human review.

Limitations and Future Improvements

- Synthetic training data — Models trained on generated data. Real world accuracy would require production complaint data for fine-tuning.
- Confidence calibration — A production system would use Platt scaling for better calibrated confidence scores.
- Officer availability — Current workload is static. A production system would update workload in real time from a database.
- Model retraining — No automated retraining pipeline. Production would retrain periodically on new resolved complaints.


Tech Stack
| Component | Technology |
| --- | --- |
| Embeddings | sentence-transformers (offline) |
| Vector Search | FAISS |
| Classification | scikit-learn Random Forest |
| Regression | scikit-learn Random Forest |
| Speech-to-Text | OpenAI Whisper (offline) |
| Translation | deep-translator |
| API Framework | FastAPI |
| Data Generation | Faker |
