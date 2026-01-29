# 🛡️ Video Classifier - Harmful Content Detection for Children

Hệ thống phát hiện video độc hại cho trẻ em sử dụng **Multimodal Learning** (kết hợp hình ảnh và văn bản).

## 📋 Tổng quan

Phân loại video thành 4 loại:
- ✅ **Safe** - Nội dung an toàn
- ⚠️ **Aggressive** - Bạo lực, ngôn từ thô tục
- 🔞 **Sexual** - Nội dung khiêu dâm
- 🔮 **Superstition** - Mê tín dị đoan

## 🏗️ Kiến trúc

```
Video → [Frame Extraction] → Image Encoder (EfficientNet-B0)  ─┐
                                                               ├→ Attention Pooling → Gated Fusion → Classification
      → [Whisper ASR]      → Text Encoder (MiniLM)           ─┘
```

**Tech Stack:**
- **Backend:** FastAPI, Apache Airflow, Apache Spark
- **Frontend:** React.js, Vite
- **Storage:** PostgreSQL, MinIO (S3)
- **Queue:** Apache Kafka
- **ML:** PyTorch, Transformers, Whisper

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/tiennguyen0000/KidSafeVideo-Detector.git
cd KidSafeVideo-Detector
```

### 2. Cấu hình Environment Variables

Tạo file `.env` từ template hoặc copy nội dung bên dưới:

```bash
# PostgreSQL Configuration
POSTGRES_USER=video_classifier
POSTGRES_PASSWORD=changeme123
POSTGRES_DB=video_classifier
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_BUCKET=video-storage
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Airflow Configuration
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://video_classifier:changeme123@postgres:5432/video_classifier
AIRFLOW__CORE__FERNET_KEY=ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg=
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__CORE__PARALLELISM=2
AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG=1
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG=1

# Model Configuration
MODEL_MODE=ultra_light  # 'ultra_light' or 'balanced'

# YouTube API Key (required for video search)
YOUTUBE_API_KEY=your_youtube_api_key_here

# Groq API Keys (required for Whisper transcription via Groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_API_KEYS=key1,key2,key3  # Multiple keys for rate limiting

USE_SPARK=true
```

> ⚠️ **Quan trọng:** Cần thay đổi `YOUTUBE_API_KEY` và `GROQ_API_KEY` bằng API keys thật của bạn.

### 3. Chuẩn bị Dataset

#### Cấu trúc thư mục `data/raw/`:

```
data/
└── raw/
    ├── labels.csv          # File metadata chính (hoặc labels1.csv, labels2.csv,...)
    └── videos/
        ├── Aggressive/     # Videos thuộc nhóm Aggressive
        │   ├── video1.mp4
        │   └── video2.mp4
        ├── Safe/           # Videos thuộc nhóm Safe  
        │   └── video3.mp4
        ├── Sexual/         # Videos thuộc nhóm Sexual
        │   └── video4.mp4
        └── Superstition/   # Videos thuộc nhóm Superstition
            └── video5.mp4
```

#### Format file CSV (labels.csv):

| Column | Mô tả | Bắt buộc |
|--------|-------|----------|
| `filename` | Đường dẫn tương đối đến video (vd: `Safe/video1.mp4`) | ❌ |
| `link` | URL gốc của video (YouTube/TikTok) - dùng làm ID duy nhất | ✅ |
| `category_real` | Label: `Safe`, `Aggressive`, `Sexual`, `Superstition` | ✅ |
| `title` | Tiêu đề video | ❌ |
| `speech2text` | Transcript có sẵn (nếu có) | ❌ |

**Ví dụ:**
```csv
filename,title,link,category_real
Aggressive/7481277306493712831.mp4,Video title,https://...,Aggressive
Safe/1234567890.mp4,Safe video,https://...,Safe
```

#### Cách thức hoạt động:

1. **Nếu có file video local:** Đặt videos vào `data/raw/videos/` theo cấu trúc `{Label}/{video_id}.mp4`
2. **Nếu chỉ có URL:** Hệ thống sẽ tự động download từ YouTube/TikTok
3. **Ingest nhiều file CSV:** Có thể tạo nhiều file CSV (labels1.csv, labels2.csv,...) và ingest riêng từng file

### 4. Khởi chạy Services

```bash
# Build và start tất cả containers
docker-compose up -d --build

# Xem logs
docker-compose logs -f

# Kiểm tra trạng thái
docker-compose ps
```

**Thời gian khởi động:** ~2-5 phút (lần đầu cần build Docker images)

### 5. Ingest Dataset

#### Cách 1: Qua Airflow UI

1. Truy cập Airflow: http://localhost:8080 (admin/admin)
2. Enable DAG `data_ingestion_dag`
3. Trigger DAG với config (optional):
   ```json
   {"csv_path": "/opt/airflow/data/raw/labels.csv"}
   ```

#### Cách 2: Qua API

```bash
# Ingest từ file CSV mặc định
curl -X POST http://localhost:8001/api/training/trigger \
  -H "Content-Type: application/json" \
  -d '{"run_ingestion": true, "run_preprocessing": true}'

# Ingest từ file CSV cụ thể
curl -X POST http://localhost:8001/api/ingestion/trigger \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/opt/airflow/data/raw/labels2.csv"}'
```

### 6. Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | - |
| **API Docs** | http://localhost:8001/docs | - |
| **Airflow** | http://localhost:8080 | admin / admin |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |

### 7. Workflow cơ bản

```
1. Ingest Data     → Upload videos + metadata vào hệ thống
2. Preprocessing   → Extract frames + Whisper transcript → Embeddings
3. Training        → Train model với Gated Fusion
4. Inference       → Predict video mới
```

> 💡 **Tips:** Sau khi ingest data mới, hệ thống có thể tự động chạy preprocessing và training nếu cấu hình `auto_train: true`.

## 📁 Cấu trúc

```
ct3/
├── backend/
│   ├── api/              # FastAPI endpoints
│   ├── airflow/dags/     # Airflow DAGs
│   └── common/
│       ├── models/       # ML models (fusion, encoders)
│       ├── pipelines/    # Training & Inference
│       └── io/           # Database, Storage, Kafka
├── frontend/             # React.js UI
├── docker/               # Dockerfiles
├── config/               # Configuration
└── data/raw/             # Dataset
```

## 🔧 Chế độ hoạt động

| Mode | Image Encoder | Text Encoder | RAM | GPU |
|------|--------------|--------------|-----|-----|
| **Ultra-Light** | EfficientNet-B0 | MiniLM | 16GB | ❌ |
| **Balanced** | ResNet50 | PhoBERT | 32GB | ✅ |

## 📊 Kết quả

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Gated Fusion (Ultra-Light) | 82.5% | 0.81 |
| Gated Fusion (Balanced) | 85.1% | 0.84 |

## 🔄 Pipeline

1. **Data Ingestion:** Upload CSV → Download videos → Store to MinIO
2. **Preprocessing:** Extract frames (16) + Whisper transcript → Embeddings
3. **Training:** Attention Pooling + Gated Fusion → Model
4. **Inference:** Video → Preprocessing → Predict → Results

## 📝 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/health` | GET | Health check |
| `/api/search` | POST | Search YouTube/TikTok |
| `/api/inference` | POST | Run inference |
| `/api/training/trigger` | POST | Trigger training |

