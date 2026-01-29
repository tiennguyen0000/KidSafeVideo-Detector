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
git clone <repo>
cd ct3
cp .env.example .env  # Configure environment variables
```

### 2. Start Services
```bash
docker-compose up -d
```

### 3. Access
- **Frontend:** http://localhost:3000
- **API:** http://localhost:8000
- **Airflow:** http://localhost:8080 (admin/admin)
- **MinIO:** http://localhost:9001

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

