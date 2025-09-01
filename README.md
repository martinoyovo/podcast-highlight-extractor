# 🧠 Podcast Highlight Extractor

An intelligent AI-powered tool that automatically extracts the most important moments from podcast transcripts using advanced Natural Language Processing (NLP).

## ✨ Features

- **Automatic Highlight Detection**: Uses machine learning to identify key moments
- **Multi-language Support**: French and English transcripts
- **Sentiment Analysis**: Analyzes emotional tone of content
- **Entity Recognition**: Identifies important people, places, and concepts
- **Confidence Scoring**: Provides reliability metrics for each highlight
- **Export Options**: JSON export for further processing
- **Beautiful UI**: Modern, responsive Gradio interface

## 🚀 Live Demo

Deployed on Render: [Your App URL will appear here after deployment]

## 🛠️ Technology Stack

- **Backend**: Python 3.9+
- **NLP Libraries**: NLTK, spaCy, scikit-learn
- **Web Framework**: Gradio
- **Deployment**: Render
- **ML Features**: TF-IDF, Sentiment Analysis, Entity Recognition

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/podcast-highlight-extractor.git
   cd podcast-highlight-extractor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLP models**
   ```bash
   python -m spacy download fr_core_news_sm
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:7860`

## 🎯 How to Use

1. **Paste your transcript** into the text area
2. **Configure settings**:
   - Choose language (French/English)
   - Set number of highlights (1-15)
   - Adjust confidence threshold
3. **Click "Analyser et Extraire"**
4. **Review results**:
   - See extracted highlights with confidence scores
   - View detailed analytics
   - Export JSON data

## 📊 How It Works

The system uses a hybrid approach combining:

1. **Keyword Detection**: Identifies important terms and phrases
2. **TF-IDF Analysis**: Measures term importance across the document
3. **Sentiment Analysis**: Evaluates emotional content
4. **Entity Recognition**: Finds named entities and concepts
5. **Statistical Scoring**: Combines multiple factors for ranking

## 🚀 Deployment

### Render Deployment

This app is configured for automatic deployment on Render:

1. **Connect your GitHub repository** to Render
2. **Create a new Web Service**
3. **Configure build settings**:
   - Build Command: `pip install -r requirements.txt && python -m spacy download fr_core_news_sm`
   - Start Command: `python app.py`
4. **Set environment variables**:
   - `PORT`: 7860
   - `PYTHON_VERSION`: 3.9.16

## 📝 Example Transcripts

The app includes sample transcripts for testing:
- **Podcast IA**: AI and technology discussion
- **Interview Business**: Entrepreneurship insights
- **Cours Technique**: Technical tutorial content

## 🔧 Configuration

### Parameters

- **Language**: Optimizes analysis for French or English
- **Max Highlights**: Number of top moments to extract (1-15)
- **Confidence Threshold**: Minimum quality score (0.1-1.0)

### Advanced Features

- **Sentiment Distribution**: Shows positive/negative/neutral content ratio
- **Entity Analysis**: Counts important named entities
- **Keyword Extraction**: Identifies most significant terms
- **Time Estimation**: Approximates highlight positions in audio

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NLTK**: Natural language processing toolkit
- **spaCy**: Industrial-strength NLP library
- **scikit-learn**: Machine learning library
- **Gradio**: Web interface framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the example transcripts

---

**Made with ❤️ for the podcast community**
