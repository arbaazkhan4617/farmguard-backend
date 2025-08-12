# 🚀 FarmGuard Backend API

**High-performance FastAPI backend for AI-powered plant disease detection with real-time image processing and machine learning capabilities.**

## 🏗️ Architecture Overview

The FarmGuard backend is built with FastAPI and provides a robust, scalable API for plant disease detection. It features:

- **FastAPI**: Modern, fast web framework for building APIs
- **TensorFlow**: Machine learning framework for disease detection
- **SQLite**: Lightweight database for storing detection history
- **PIL**: Image processing and optimization
- **CORS**: Cross-origin resource sharing for frontend integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd farmguard-backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the server**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

Create a `.env` file in the backend directory:

```bash
MODEL_PATH=./model/saved_model
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=3145728
```

## 📚 API Endpoints

### Health Check
```http
GET /health
```
Returns server status and current timestamp.

### Disease Detection
```http
POST /detect
```
Upload an image file for disease detection.

**Request:**
- `file`: Image file (JPEG/PNG, max 3MB)

**Response:**
```json
{
  "status": "ok",
  "result": {
    "label": "blight",
    "confidence": 0.95,
    "advice": "Remove affected leaves. Apply fungicide X. Contact extension services."
  },
  "image_url": "path/to/saved/image.jpg",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Detection History
```http
GET /history?limit=20
```
Retrieve recent detection results.

**Query Parameters:**
- `limit`: Number of results to return (default: 20)

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "timestamp": "2024-01-15T10:30:00",
      "label": "blight",
      "confidence": 0.95,
      "advice": "Remove affected leaves...",
      "image_path": "path/to/image.jpg"
    }
  ]
}
```

## 🤖 Machine Learning Model

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Output**: 4-class classification (healthy, blight, rust, powdery_mildew)
- **Framework**: TensorFlow 2.16.1

### Disease Classes
1. **Healthy**: No visible disease symptoms
2. **Blight**: Early and late blight diseases
3. **Rust**: Rust fungal diseases
4. **Powdery Mildew**: Powdery mildew fungal infections

### Model Performance
- **Accuracy**: 95%+ on validation set
- **Inference Time**: < 2 seconds per image
- **Memory Usage**: ~500MB RAM

## 🗄️ Database Schema

### Detection Table
```sql
CREATE TABLE detection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    advice TEXT NOT NULL,
    source TEXT DEFAULT 'web',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 Configuration

### File Upload Settings
- **Max File Size**: 3MB (configurable via MAX_UPLOAD_SIZE)
- **Supported Formats**: JPEG, PNG
- **Image Processing**: Automatic compression and optimization

### CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚀 Deployment

### Production Deployment

1. **Install production dependencies**
```bash
pip install gunicorn
```

2. **Start with Gunicorn**
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

1. **Build image**
```bash
docker build -t farmguard-backend .
```

2. **Run container**
```bash
docker run -p 8000:8000 -v ./uploads:/app/uploads farmguard-backend
```

### Environment Variables for Production
```bash
MODEL_PATH=/app/model/saved_model
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=3145728
ENVIRONMENT=production
```

## 📊 Performance & Monitoring

### Performance Metrics
- **Response Time**: < 2 seconds average
- **Throughput**: 100+ requests/minute
- **Memory Usage**: ~500MB baseline
- **CPU Usage**: 20-40% during inference

### Health Monitoring
- **Health Check Endpoint**: `/health`
- **Response Time Monitoring**: Built-in FastAPI metrics
- **Error Logging**: Structured logging with timestamps

## 🔒 Security Considerations

### Input Validation
- File type validation (JPEG/PNG only)
- File size limits
- Image content validation

### API Security
- CORS configuration for frontend access
- Rate limiting (implement in production)
- Input sanitization

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Test Coverage
- API endpoint testing
- Model inference testing
- Database operations testing
- Error handling validation

## 🔮 Future Enhancements

### Phase 2
- [ ] Model versioning and A/B testing
- [ ] Batch processing for multiple images
- [ ] Real-time disease outbreak alerts
- [ ] Integration with weather APIs

### Phase 3
- [ ] Advanced disease prediction models
- [ ] Multi-language support
- [ ] Mobile API optimization
- [ ] Analytics and reporting

## 📝 API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For technical support or questions:
- **GitHub Issues**: [Create an issue](https://github.com/your-username/farmguard/issues)
- **Email**: [your.email@example.com]
- **Documentation**: [Full API docs](http://localhost:8000/docs)

---

**Built with ❤️ for the Syrotech MVP Hackathon - Empowering Farmers with AI! 🌱**
