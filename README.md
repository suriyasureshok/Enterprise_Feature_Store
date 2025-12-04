# Enterprise Feature Store

A comprehensive feature store platform for enterprise machine learning applications that combines event analytics, NLP-based review sentiment analysis, and churn prediction. This system transforms raw customer data and textual feedback into reusable, production-ready features for multiple downstream ML models.

## Overview

The Enterprise Feature Store is designed to:
- **Unify Data**: Consolidate customer profiles, behavioral events, and review data into a single feature repository
- **Enrich Features**: Apply advanced NLP techniques (sentiment analysis, topic modeling) and behavioral analytics
- **Serve Real-time**: Provide low-latency feature access through Redis-based online store
- **Enable Scalability**: Support batch and real-time feature computation for enterprise-scale operations

## Key Features

- **Event Analytics Pipeline**: Process user activity events (purchases, reviews, cart actions) to derive behavioral features
- **NLP Review Processing**: Extract sentiment scores, topic vectors, and text embeddings using VADER sentiment analysis and LDA topic modeling
- **Offline/Online Stores**: Parquet-based offline storage for training datasets and Redis online store for real-time inference
- **Feature Registry**: YAML-based feature definitions and metadata management
- **Churn Prediction**: End-to-end ML pipeline for predicting customer churn with multiple model algorithms

## Project Structure

```
.
├── feature_store/                    # Core feature store module
│   ├── feature_pipeline/             # Feature engineering pipelines
│   │   ├── user_activity_features.py     # Behavioral features from events
│   │   ├── user_profile_features.py      # Demographic and profile features
│   │   ├── user_review_features.py       # NLP features from reviews (sentiment, topics)
│   │   └── run_feature_pipelines.py      # Orchestration script
│   ├── offline_store/                # Batch feature storage (Parquet)
│   ├── online_store/                 # Real-time feature store (Redis)
│   │   └── push_features_to_online_store.py
│   ├── transformations/              # Data transformation utilities
│   ├── utils/                        # Helper functions
│   └── feature_registry.yaml         # Feature definitions and metadata
├── models/                           # ML models
│   ├── training/                     # Model training code
│   └── inference/                    # Model inference code
├── services/                         # API services
│   ├── feature_apis/                 # Feature retrieval APIs
│   └── model_apis/                   # Model prediction APIs
├── notebooks/                        # Jupyter notebooks
│   ├── data_synthesis.ipynb          # Synthetic data generation
│   └── model_training.ipynb          # Model training and evaluation
├── data/                             # Data directory
│   ├── raw/                          # Raw data files
│   ├── processed/                    # Processed data
│   └── schema/                       # Data schemas
├── pyproject.toml                    # Python project configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- Redis (for online store)
- pip or uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/suriyasureshok/Enterprise_Feature_Store.git
cd Enterprise_Feature_Store
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# or
uv add -r requirements.txt
```

4. Start Redis (required for online store):
```bash
redis-server
```

### Running Feature Pipelines

Generate features from raw data:
```bash
python -m feature_store.feature_pipeline.run_feature_pipelines
```

This will:
- Load sample data from `data/processed/`
- Extract user activity features
- Extract user profile features
- Extract user review features (with sentiment and topic vectors)
- Save aggregated features to `offline_store/user_*_features.parquet`

### Pushing Features to Online Store

Deploy features to Redis for real-time serving:
```bash
python -m online_store.push_features_to_online_store
```

### Model Training

Train churn prediction models in the notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

Or run via CLI:
```bash
python -m models.training.train_churn_model
```

## Data Flow

```
Raw Data (CSV)
    ↓
Feature Extraction Pipelines
    ├─ Event Analytics (user_activity_features.py)
    ├─ Profile Features (user_profile_features.py)
    └─ NLP Features (user_review_features.py)
    ↓
Offline Store (Parquet)
    ↓
Feature Materialization
    ↓
Online Store (Redis) ← Real-time Inference
    ↓
Training Dataset
    ↓
Model Training
    ↓
Predictions
```

## Feature Categories

### Activity Features
- `total_events`: Count of all user events
- `purchase_count`: Number of purchase events
- `add_to_cart_count`: Items added to cart
- `session_count`: Number of user sessions
- `avg_session_length`: Average session duration
- `last_active_days`: Days since last activity
- `event_vector_*`: One-hot encoded event types

### Profile Features
- `age`: User age
- `gender`: User gender (one-hot encoded)
- `premium`: Premium membership status
- `loyalty_status`: Loyalty tier (Gold/Silver/Regular, one-hot encoded)
- `device`: Device type (one-hot encoded)
- `location`: Geographic location (one-hot encoded)
- `education`: Education level (one-hot encoded)

### Review Features
- `avg_rating`: Average review rating
- `total_reviews`: Total reviews written
- `avg_text_length`: Average review length
- `avg_sentiment_score`: VADER sentiment scores (compound)
- `topic_vector`: LDA topic distribution (10 topics)

## Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **NLP**: NLTK (VADER), spaCy, sentence-transformers
- **Feature Store**: Parquet (offline), Redis (online)
- **APIs**: FastAPI
- **Notebooks**: Jupyter

## Configuration

### Feature Registry (`feature_registry.yaml`)
Define and document features in YAML format for discovery and governance.

### Environment Variables
Create a `.env` file for sensitive configurations:
```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## Known Issues & Solutions

See `issues_encountered.md` for a list of issues encountered during development and their resolutions.

## Contributing

1. Create a feature branch
2. Make your changes
3. Test with `pytest`
4. Submit a pull request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For questions or support, contact: suriyasureshok@example.com

## Roadmap

- [ ] Implement feature monitoring and drift detection
- [ ] Add support for streaming feature computation (Kafka/Flink)
- [ ] Implement feature versioning and time-travel queries
- [ ] Add advanced imputation strategies for missing values
- [ ] Expand NLP models (BERT-based embeddings, advanced topic modeling)
- [ ] Build dashboard for feature discovery and lineage tracking
