# Enterprise Feature Store - Architecture

## System Architecture Overview

The Enterprise Feature Store is built on a modular, scalable architecture that supports both batch and real-time feature computation. The system is organized into distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                               │
│  (Customer Data, Events, Reviews)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Pipeline Layer                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │   Activity       │  │   Profile        │  │   Review (NLP)   │
│  │   Features       │  │   Features       │  │   Features       │
│  │   Pipeline       │  │   Pipeline       │  │   Pipeline       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Offline Store Layer                           │
│              (Parquet - Batch Storage)                          │
│  • user_activity_features.parquet                               │
│  • user_profile_features.parquet                                │
│  • user_review_features.parquet                                 │
│  • training_dataset.parquet (materialized)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────┬─────────────────┐
        ↓                                       ↓                 ↓
┌──────────────────┐               ┌──────────────────┐  ┌────────────┐
│   Online Store   │               │  Model Training  │  │  Features  │
│   (Redis)        │               │  Pipeline        │  │  API       │
│                  │               │                  │  │  (FastAPI) │
│ Real-time        │               │  • Data Prep     │  │            │
│ Feature Serving  │               │  • Model Fit     │  │ GET /user/ │
│                  │               │  • Evaluation    │  │ {user_id}  │
└──────────────────┘               └──────────────────┘  └────────────┘
        ↑                                       ↓
        │                          ┌──────────────────┐
        │                          │  ML Models       │
        │                          │  • Logistic Reg. │
        │                          │  • XGBoost       │
        │                          │  • Random Forest │
        │                          └──────────────────┘
        │                                   ↓
        └─────────────────────────────────────┘
              Real-time Inference API
              (model_apis)
```

## Component Details

### 1. Feature Pipeline Layer

The feature pipeline layer is organized into three specialized pipelines:

#### **Activity Features Pipeline** (`user_activity_features.py`)
- **Input**: Event logs (purchases, cart actions, page views, etc.)
- **Processing**:
  - Aggregates events by user and event type
  - Computes behavioral metrics (counts, sessions, durations)
  - Calculates recency (days since last activity)
  - Creates one-hot encoded event type vectors
- **Output**: `user_activity_features.parquet`
- **Key Features**:
  - `total_events`: Count of all events
  - `purchase_count`, `add_to_cart_count`, `session_count`
  - `avg_session_length`: Minutes per session
  - `last_active_days`: Recency metric
  - `event_vector_*`: Categorical event types

#### **Profile Features Pipeline** (`user_profile_features.py`)
- **Input**: Customer demographic and profile data
- **Processing**:
  - One-hot encodes categorical features (gender, loyalty_status, device, location, education)
  - Ensures consistent feature encoding across batches
  - Handles missing values with appropriate defaults
- **Output**: `user_profile_features.parquet`
- **Key Features**:
  - `age`, `gender`, `premium`
  - `loyalty_status`, `device`, `location`, `education` (encoded)

#### **Review Features Pipeline** (`user_review_features.py`)
- **Input**: Customer review texts and metadata
- **Processing**:
  - **Sentiment Analysis**: VADER sentiment intensity analyzer computes compound sentiment scores
  - **Topic Modeling**: Latent Dirichlet Allocation (LDA) with 10 topics
  - **Text Preprocessing**: Tokenization, stopword removal, lemmatization
  - Handles null/missing review text gracefully
- **Output**: `user_review_features.parquet`
- **Key Features**:
  - `avg_rating`: Mean review rating
  - `total_reviews`: Count of reviews
  - `avg_sentiment_score`: VADER compound sentiment (-1 to 1)
  - `topic_vector`: 10-dimensional LDA topic distribution

### 2. Offline Store Layer

**Technology**: Apache Parquet (columnar, compressed format)

**Purpose**: Efficient batch storage for historical features and training datasets

**Components**:
- `offline_store/user_activity_features.parquet` - Activity features for all users
- `offline_store/user_profile_features.parquet` - Profile features for all users
- `offline_store/user_review_features.parquet` - Review features for all users
- `materialized_store/training_dataset.parquet` - Joined, materialized dataset for model training

**Advantages**:
- Efficient compression and column-oriented storage
- Fast batch reads for ML training
- Time-travel queries possible with partitioning
- Cost-effective for large-scale storage

### 3. Online Store Layer

**Technology**: Redis (in-memory key-value store)

**Purpose**: Real-time, low-latency feature serving for inference

**Data Structure**:
```
Key: user:{user_id}
Value: Hash containing all user features
  - avg_sentiment_score: "0.45"
  - premium: "1"
  - topic_vector_0: "0.12"
  - topic_vector_1: "0.08"
  - ... (all feature columns)
```

**Serialization**:
- Numeric features: Direct serialization
- Arrays/vectors: JSON serialization with truncation/padding to fixed length
- Booleans: String representation ("True"/"False")
- Null values: Empty string ("")

**Deployment**:
```bash
python -m online_store.push_features_to_online_store
```

Pushes all materialized features to Redis in hash format for real-time access.

### 4. Model Training Pipeline

**Technology**: Scikit-learn, XGBoost, custom training framework

**Process**:
1. **Data Preparation**:
   - Load materialized training dataset
   - Expand list-type columns (e.g., topic_vector) into fixed-length numeric columns
   - One-hot encode remaining categorical features
   - Select numeric columns only for model compatibility

2. **Train/Test Split**:
   - 80/20 stratified split preserving class distribution
   - Ensures balanced churn representation

3. **Model Training**:
   - **Logistic Regression**: Baseline model with L2 regularization
   - **XGBoost**: Gradient boosting for improved performance
   - **Random Forest**: Ensemble alternative with feature importance
   - Hyperparameter tuning via cross-validation

4. **Evaluation**:
   - Classification report: Precision, recall, F1-score
   - ROC-AUC score for ranking quality
   - Feature importance analysis

### 5. APIs and Services

#### **Feature Serving API** (`services/feature_apis/`)
- **Endpoint**: `GET /features/{user_id}`
- **Response**: All features for a user from Redis
- **Purpose**: Real-time feature lookup for inference

#### **Model Prediction API** (`services/model_apis/`)
- **Endpoint**: `POST /predict/churn`
- **Input**: User ID or raw features
- **Response**: Churn probability and prediction
- **Purpose**: Real-time inference server

#### **Orchestration**
- FastAPI-based REST services
- Deployed with Uvicorn
- Stateless, horizontally scalable

## Data Flow - Detailed

### Batch Feature Generation (Daily/Scheduled)
```
1. Raw Data Load
   ↓
2. Feature Pipeline Execution (parallel)
   ├─ Activity: Aggregate events → Activity features
   ├─ Profile: Transform demographics → Profile features
   └─ Reviews: NLP processing → Review features
   ↓
3. Feature Aggregation
   - Join all features by user_id
   ↓
4. Materialization
   - Create training_dataset.parquet
   ↓
5. Offline Store
   - Write individual feature tables (Parquet)
   ↓
6. Online Store Sync
   - (Optional) Push new features to Redis
```

### Real-time Serving
```
User Request
   ↓
Feature API
   ├─ Query Redis by user_id
   ├─ Return hash of features
   ↓
Model API
   ├─ Receive features
   ├─ Load trained model
   ├─ Generate prediction
   ↓
Response (Churn probability)
```

## Feature Registry

**File**: `feature_store/feature_registry.yaml`

**Purpose**: 
- Centralized feature metadata management
- Feature discovery
- Data lineage tracking
- Version management

**Structure**:
```yaml
features:
  - name: avg_sentiment_score
    description: Average VADER sentiment score of user reviews
    owner: nlp_team
    type: float
    pipeline: review_features
    version: 1.0
  - name: total_reviews
    description: Total count of reviews by user
    owner: analytics_team
    type: int
    pipeline: review_features
    version: 1.0
```

## Scalability Considerations

### Horizontal Scaling
- **Feature Pipelines**: Distribute event processing across multiple workers
- **Redis**: Cluster mode for distributed online store
- **APIs**: Multiple FastAPI instances behind load balancer

### Batch Optimization
- Pandas vectorized operations (avoid Python loops)
- Parquet partitioning by date or user cohorts
- Efficient memory management for large datasets

### Real-time Optimization
- Redis clustering and replication
- Connection pooling
- Caching strategies

## Failure Modes & Recovery

| Component | Failure | Recovery |
|-----------|---------|----------|
| Offline Store | Corruption | Recreate from raw data pipelines |
| Online Store (Redis) | Downtime | Failover to secondary Redis instance |
| Feature Pipelines | Errors | Restart pipeline, validate data quality |
| APIs | Crash | Health checks + auto-restart |
| Model | Degradation | Monitor prediction quality, retrain |

## Deployment Architecture (Recommended)

```
Local Development:
  - Feature pipelines in Python scripts
  - Redis local instance
  - Jupyter for experimentation

Staging:
  - Orchestrated pipelines (Airflow/Dagster)
  - Redis Sentinel
  - Containerized services (Docker)

Production:
  - Kubernetes orchestration
  - Redis Cluster
  - Multi-zone Redis replicas
  - Load-balanced API services
  - Monitoring (Prometheus/Grafana)
  - Logging (ELK stack)
```

## Performance Metrics

- **Feature Pipeline Latency**: ~5-10 minutes for full batch
- **Online Store Lookup**: <10ms per user from Redis
- **API Response Time**: <100ms (including model inference)
- **Storage**: ~100MB per 10K users (compressed Parquet)
- **Redis Memory**: ~1MB per 1000 users

## Security Considerations

- Encrypt Redis connections (TLS)
- API authentication (JWT tokens)
- Row-level access control for sensitive features
- Audit logging for feature access
- Data minimization (remove PII after feature extraction)

## Future Enhancements

1. **Feature Versioning**: Time-travel queries for historical features
2. **Streaming Pipeline**: Kafka → Feature computation → Redis
3. **Feature Monitoring**: Drift detection, data quality checks
4. **Advanced NLP**: BERT embeddings, multi-lingual support
5. **Graph Features**: User-user relationships, item embeddings
6. **AutoML Integration**: Automated model selection and hyperparameter tuning
