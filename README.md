# Podcast Highlight Extractor

> **AI-Powered NLP Tool for Automatic Podcast Highlight Extraction**

An intelligent system that automatically identifies and extracts the most important moments from podcast transcripts using advanced Natural Language Processing techniques including TF-IDF analysis, sentiment analysis, and entity recognition.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-green.svg)](https://spacy.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- **🚀 Automatic Highlight Detection**: ML-powered identification of key moments
- **🌍 Multi-language Support**: French and English with optimized models
- **📊 Sentiment Analysis**: Emotional tone analysis using NLTK VADER
- **🔍 Entity Recognition**: Named entity detection with spaCy
- **📈 TF-IDF Scoring**: Term importance analysis for better ranking
- **🎯 Confidence Scoring**: Reliability metrics for each highlight
- **💾 Export Options**: JSON export for further processing
- **🎨 Modern UI**: Beautiful, responsive Gradio interface
- **☁️ Cloud Ready**: Deployable on Render, Heroku, or any cloud platform

## Live Demo

**Deployed on Render**: https://podcast-highlight-extractor.onrender.com

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   Python Backend │    │   NLP Pipeline  │
│                 │    │                  │    │                 │
│ • Input Forms   │◄──►│ • Highlight      │◄──►│ • TF-IDF        │
│ • Results       │    │   Extractor      │    │ • Sentiment     │
│ • Examples      │    │ • API Endpoints  │    │ • Entities      │
│ • Export        │    │ • Error Handling │    │ • Scoring       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Technology Stack

### **Core Technologies**
- **Backend**: Python 3.9+
- **Web Framework**: Gradio 4.0+
- **Deployment**: Render (with render.yaml)

### **NLP Libraries**
- **spaCy**: Entity recognition & sentence segmentation
- **NLTK**: Sentiment analysis & stop words
- **scikit-learn**: TF-IDF vectorization & ML algorithms

### **Models Used**
- **French**: `fr_core_news_sm-3.8.0` (16.3 MB)
- **English**: Default spaCy English model
- **Sentiment**: VADER Lexicon (NLTK)

## Quick Start

### **Prerequisites**
- Python 3.9 or higher
- 2GB+ RAM (for NLP models)
- Internet connection (for model downloads)

### **1. Clone & Setup**
```bash
git clone https://github.com/yourusername/podcast-highlight-extractor.git
cd podcast-highlight-extractor
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download NLP Models**
```bash
# French model (required for French support)
python -m spacy download fr_core_news_sm

# NLTK resources
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
```

### **4. Run the Application**
```bash
python app.py
```

### **5. Open Your Browser**
Navigate to `http://localhost:7860`

## How to Use

### **Basic Usage**
1. **Paste your transcript** into the text area
2. **Configure settings**:
   - Choose language (French/English)
   - Set number of highlights (1-15)
   - Adjust confidence threshold (0.1-1.0)
3. **Click "Analyser et Extraire"** (or "Analyze and Extract")
4. **Review results** with confidence scores and analytics

### **Example Transcripts**
The app includes sample content for testing:
- **Podcast IA**: AI and technology discussion (French)
- **Interview Business**: Entrepreneurship insights (French)
- **Cours Technique**: Technical tutorial on microservices (French)
- **🌟 AI Enthusiasm**: Highly positive AI content (English)

### **Output Format**
```json
{
  "highlights": [
    {
      "sentence_id": 0,
      "text": "Your highlighted text here",
      "confidence": 0.85,
      "importance_score": 0.72,
      "start_time": 0,
      "end_time": 30,
      "features": {
        "sentiment": 0.45,
        "entities": 2,
        "keywords": 3,
        "tfidf_score": 0.67
      }
    }
  ],
  "metadata": {
    "total_sentences": 15,
    "processing_method": "hybrid_nlp_prod",
    "language": "fr",
    "version": "2.0"
  }
}
```

## Configuration

### **Environment Variables**
```bash
# Optional: Custom port
export PORT=8080

# Optional: Python version
export PYTHON_VERSION=3.9.16
```

### **Customization Options**
- **Language Models**: Add new spaCy models for other languages
- **Keyword Patterns**: Modify regex patterns in `extract_features()`
- **Scoring Weights**: Adjust weights in the composite scoring algorithm
- **UI Theme**: Customize Gradio theme and CSS

## Deployment

### **Render (Recommended)**
1. **Connect your GitHub repo** to Render
2. **Create new Web Service**
3. **Render automatically detects** your `render.yaml`
4. **Deploy** - everything is pre-configured!

### **Other Platforms**
```bash
# Heroku
heroku create your-app-name
git push heroku main

# Docker
docker build -t highlight-extractor .
docker run -p 7860:7860 highlight-extractor

# Local Production
gunicorn -w 4 -b 0.0.0.0:7860 app:app
```

## Performance & Benchmarks

### **Processing Speed**
- **Small transcripts** (<500 words): ~2-3 seconds
- **Medium transcripts** (500-2000 words): ~5-8 seconds
- **Large transcripts** (>2000 words): ~10-15 seconds

### **Accuracy Metrics**
- **English Content**: 91.7% positive sentiment detection
- **French Content**: 92.9% neutral sentiment detection
- **Entity Recognition**: 1-3 entities per highlight (English)
- **Highlight Relevance**: Top-ranked highlights consistently relevant

### **Resource Usage**
- **Memory**: ~500MB (with NLP models loaded)
- **CPU**: Single-threaded, optimized for cloud deployment
- **Storage**: ~100MB (models + application)

## How It Works

### **1. Text Preprocessing**
```
Raw Text → Clean Text → Sentence Segmentation → Feature Extraction
```

### **2. Feature Extraction**
- **Keywords**: Regex patterns for important terms
- **Sentiment**: VADER sentiment analysis
- **Entities**: Named entity recognition (spaCy)
- **TF-IDF**: Term frequency-inverse document frequency

### **3. Scoring Algorithm**
```
Final Score = Σ(Weight × Normalized_Feature)

Weights:
- Importance Keywords: 30%
- TF-IDF Score: 25%
- Sentiment: 15%
- Entity Count: 10%
- Questions/Exclamations: 15%
- Numbers: 5%
```

### **4. Highlight Selection**
- Rank sentences by composite score
- Apply confidence threshold
- Return top N highlights with metadata

## Testing

### **Run Tests**
```bash
# Basic functionality test
python -c "from app import HighlightExtractor; e = HighlightExtractor('en'); print('Basic test passed')"

# Performance test
python -c "
import time
from app import HighlightExtractor
e = HighlightExtractor('en')
start = time.time()
result = e.extract_highlights('Your test text here...')
print(f'Processing time: {time.time() - start:.2f}s')
"
```

### **Test Data**
- **French**: Technical content, business interviews
- **English**: AI discussions, educational content
- **Edge Cases**: Very short/long texts, mixed languages

## Contributing

### **Development Setup**
```bash
# Fork and clone
git clone https://github.com/yourusername/podcast-highlight-extractor.git
cd podcast-highlight-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run linting
black app.py
flake8 app.py
```

### **Contributing Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Troubleshooting

### **Common Issues**

#### **"spaCy model not found"**
```bash
python -m spacy download fr_core_news_sm
```

#### **"NLTK data not found"**
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### **"Port already in use"**
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9

# Or use different port
export PORT=8080
python app.py
```

#### **"Memory error"**
- Reduce `max_highlights` parameter
- Use smaller spaCy models
- Increase server memory (Render: upgrade plan)

### **Performance Issues**
- **Slow processing**: Check if NLP models are loaded
- **High memory**: Monitor memory usage during processing
- **Low accuracy**: Verify language model compatibility


## Acknowledgments

- **spaCy**: Industrial-strength NLP library
- **NLTK**: Natural language processing toolkit
- **scikit-learn**: Machine learning algorithms
- **Gradio**: Web interface framework
- **Render**: Cloud deployment platform

## Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/martinoyovo/podcast-highlight-extractor/issues)
- **Discussions**: [Join the community](https://github.com/martinoyovo/podcast-highlight-extractor/discussions)
- **Wiki**: [Documentation and guides](https://github.com/martinoyovo/podcast-highlight-extractor/wiki)

---

**Made with ❤️ for the podcast community**

*If this project helps you, please give it a ⭐ star on GitHub!*
