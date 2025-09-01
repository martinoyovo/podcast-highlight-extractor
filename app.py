import gradio as gr
import numpy as np
import pandas as pd
import re
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Optional imports with error handling for production
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
        # Téléchargement automatique des ressources NLTK
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_OK = True
    print("NLTK disponible")
except Exception as e:
    NLTK_OK = False
    print(f"NLTK non disponible: {e}")

try:
    import spacy
    # Test loading the French model
    try:
        nlp_fr = spacy.load('fr_core_news_sm')
        SPACY_FR_OK = True
        print("spaCy français disponible")
    except:
        SPACY_FR_OK = False
        print("spaCy français non disponible - mode basique")
except:
    SPACY_FR_OK = False
    print("spaCy non disponible - mode basique")

class HighlightExtractor:
    """Optimized highlight extractor for production"""
    
    def __init__(self, language='fr'):
        self.language = language
        print(f"🔧 Initialisation extracteur ({language})")
        
        # Configuration of stop words
        if NLTK_OK:
            try:
                if language == 'fr':
                    self.stop_words = set(stopwords.words('french'))
                else:
                    self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = self._get_basic_stopwords(language)
        else:
            self.stop_words = self._get_basic_stopwords(language)
        
        # Sentiment analyzer
        if NLTK_OK:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                print("Sentiment analyzer chargé")
            except:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        # spaCy if available
        if SPACY_FR_OK and language == 'fr':
            self.nlp = nlp_fr
            print("spaCy français chargé")
        else:
            self.nlp = None
    
    def _get_basic_stopwords(self, language):
        """Basic stop words without NLTK"""
        if language == 'fr':
            return {
                'le', 'la', 'les', 'un', 'une', 'de', 'du', 'des', 'et', 'à', 
                'ce', 'qui', 'que', 'dont', 'où', 'dans', 'sur', 'avec', 'pour',
                'par', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
                'je', 'tu', 'se', 'me', 'te', 'lui', 'leur', 'son', 'sa', 'ses',
                'est', 'sont', 'était', 'avoir', 'être', 'faire', 'dire', 'aller'
            }
        else:
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her'
            }
    
    def preprocess_transcript(self, transcript):
        """Robust preprocessing and segmentation"""
        if not transcript or not transcript.strip():
            return []
            
        # Cleaning
        text = re.sub(r'\[.*?\]', '', transcript)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 50:  # Too short
            return []
        
        # Segmentation
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]
            except:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        return sentences
    
    def extract_features(self, sentences):
        """Optimized feature extraction"""
        features = []
        
        # Multilingual patterns
        fr_patterns = [
            r'\b(important|crucial|essentiel|clé|fondamental|vital)\b',
            r'\b(découverte|révélation|surprise|incroyable|extraordinaire)\b',
            r'\b(conseil|astuce|technique|méthode|stratégie|secret)\b',
            r'\b(attention|erreur|piège|danger|warning|alerte)\b',
            r'\b(résultat|conclusion|bilan|synthèse|récapitulatif)\b',
            r'\b(innovation|nouveau|révolutionnaire|breakthrough|disruptif)\b'
        ]
        
        en_patterns = [
            r'\b(important|crucial|essential|key|fundamental|vital)\b',
            r'\b(discovery|revelation|surprise|incredible|extraordinary)\b',
            r'\b(advice|tip|technique|method|strategy|secret)\b',
            r'\b(attention|error|trap|danger|warning|alert)\b',
            r'\b(result|conclusion|summary|synthesis|recap)\b',
            r'\b(innovation|new|revolutionary|breakthrough|disruptive)\b'
        ]
        
        patterns = fr_patterns if self.language == 'fr' else en_patterns
        
        for i, sentence in enumerate(sentences):
            feature = {
                'sentence_id': i,
                'text': sentence,
                'length': len(sentence),
                'word_count': len(sentence.split()),
            }
            
            # Importance keyword score
            importance_score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, sentence.lower()))
                importance_score += matches
            
            feature['importance_keywords'] = importance_score
            
            # Pragmatic features
            feature['has_question'] = int('?' in sentence)
            feature['has_exclamation'] = int('!' in sentence)
            
            # Numbers and stats
            numbers = re.findall(r'\d+', sentence)
            feature['number_count'] = len(numbers)
            feature['has_percentage'] = int('%' in sentence or 'pourcent' in sentence.lower())
            
            # Sentiment
            if self.sentiment_analyzer:
                try:
                    scores = self.sentiment_analyzer.polarity_scores(sentence)
                    feature['sentiment_compound'] = scores['compound']
                except:
                    feature['sentiment_compound'] = 0
            else:
                feature['sentiment_compound'] = 0
            
            # Entities (basic or spaCy)
            if self.nlp:
                try:
                    doc = self.nlp(sentence)
                    feature['entity_count'] = len(doc.ents)
                except:
                    # Basic fallback
                    proper_nouns = re.findall(r'\b[A-ZÀ-Ÿ][a-zA-ZÀ-ÿ]+\b', sentence)
                    feature['entity_count'] = len(set(proper_nouns))
            else:
                proper_nouns = re.findall(r'\b[A-ZÀ-Ÿ][a-zA-ZÀ-ÿ]+\b', sentence)
                feature['entity_count'] = len(set(proper_nouns))
            
            features.append(feature)
        
        return pd.DataFrame(features)
    
    def calculate_tfidf_scores(self, sentences):
        """Robust TF-IDF calculation"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=min(200, len(sentences) * 10),
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            tfidf_scores = []
            for i in range(len(sentences)):
                row = tfidf_matrix[i].toarray()[0]
                avg_score = np.mean(row[row > 0]) if np.any(row > 0) else 0
                max_score = np.max(row)
                
                tfidf_scores.append({
                    'tfidf_avg': float(avg_score),
                    'tfidf_max': float(max_score)
                })
            
            top_terms = vectorizer.get_feature_names_out()[:10]
            return tfidf_scores, list(top_terms)
            
        except Exception as e:
            print(f"TF-IDF error: {e}")
            return [{'tfidf_avg': 0.0, 'tfidf_max': 0.0} for _ in sentences], []
    
    def extract_highlights(self, transcript, max_highlights=5, min_confidence=0.3):
        """Main extraction with robust error handling"""
        try:
            if not transcript or len(transcript.strip()) < 50:
                return {
                    'highlights': [],
                    'metadata': {'error': 'Transcript too short (minimum 50 characters)'},
                    'analytics': {}
                }
            
            # Preprocessing
            sentences = self.preprocess_transcript(transcript)
            if len(sentences) == 0:
                return {
                    'highlights': [],
                    'metadata': {'error': 'No sentences detected'},
                    'analytics': {}
                }
            
            # Feature extraction
            features_df = self.extract_features(sentences)
            
            # TF-IDF
            tfidf_scores, top_terms = self.calculate_tfidf_scores(sentences)
            
            # Add TF-IDF scores
            for i, scores in enumerate(tfidf_scores):
                if i < len(features_df):
                    features_df.loc[i, 'tfidf_avg'] = scores['tfidf_avg']
                    features_df.loc[i, 'tfidf_max'] = scores['tfidf_max']
            
            # Composite score calculation
            numeric_cols = ['importance_keywords', 'tfidf_max', 'sentiment_compound', 
                           'entity_count', 'number_count']
            
            for col in numeric_cols:
                if col in features_df.columns and len(features_df) > 1 and features_df[col].std() > 0:
                    features_df[f'{col}_norm'] = (features_df[col] - features_df[col].min()) / \
                                               (features_df[col].max() - features_df[col].min())
                else:
                    features_df[f'{col}_norm'] = 0.5  # Neutral value
            
            # Optimized weights
            weights = {
                'importance_keywords_norm': 0.30,
                'tfidf_max_norm': 0.25,
                'sentiment_compound_norm': 0.15,
                'entity_count_norm': 0.10,
                'has_question': 0.08,
                'has_exclamation': 0.07,
                'number_count_norm': 0.05
            }
            
            features_df['final_score'] = 0
            for feature, weight in weights.items():
                if feature in features_df.columns:
                    features_df['final_score'] += weight * features_df[feature]
            
            # Highlight selection
            n_candidates = min(max_highlights * 2, len(features_df))
            candidates = features_df.nlargest(n_candidates, 'final_score')
            
            highlights = []
            max_score = features_df['final_score'].max()
            min_score = features_df['final_score'].min()
            
            for _, candidate in candidates.iterrows():
                if len(highlights) >= max_highlights:
                    break
                    
                # Normalized confidence
                if max_score > min_score:
                    confidence = (candidate['final_score'] - min_score) / (max_score - min_score)
                else:
                    confidence = 0.5
                
                if confidence >= min_confidence:
                    highlight = {
                        'sentence_id': int(candidate['sentence_id']),
                        'text': str(candidate['text']),
                        'confidence': float(confidence),
                        'importance_score': float(candidate['final_score']),
                        'start_time': int(candidate['sentence_id'] * 30),
                        'end_time': int((candidate['sentence_id'] + 1) * 30),
                        'features': {
                            'sentiment': float(candidate.get('sentiment_compound', 0)),
                            'entities': int(candidate.get('entity_count', 0)),
                            'keywords': int(candidate.get('importance_keywords', 0)),
                            'tfidf_score': float(candidate.get('tfidf_max', 0))
                        }
                    }
                    highlights.append(highlight)
            
            # Results
            results = {
                'highlights': highlights,
                'metadata': {
                    'total_sentences': len(sentences),
                    'processing_method': 'hybrid_nlp_prod',
                    'top_keywords': top_terms[:10],
                    'avg_sentiment': float(features_df['sentiment_compound'].mean()),
                    'language': self.language,
                    'version': '2.0'
                },
                'analytics': {
                    'sentiment_distribution': {
                        'positive': float((features_df['sentiment_compound'] > 0.1).mean()),
                        'negative': float((features_df['sentiment_compound'] < -0.1).mean()),
                        'neutral': float((abs(features_df['sentiment_compound']) <= 0.1).mean())
                    },
                    'feature_summary': {
                        'avg_word_count': float(features_df['word_count'].mean()),
                        'questions_ratio': float(features_df['has_question'].mean()),
                        'exclamations_ratio': float(features_df['has_exclamation'].mean()),
                        'numbers_ratio': float((features_df['number_count'] > 0).mean())
                    }
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return {
                'highlights': [],
                'metadata': {'error': f'Processing error: {str(e)}'},
                'analytics': {}
            }

def process_transcript(transcript_text, language, max_highlights, min_confidence):
    """Main function for Gradio"""
    if not transcript_text or not transcript_text.strip():
        return "Please enter a transcript", "", pd.DataFrame()
    
    try:
        extractor = HighlightExtractor(language=language)
        results = extractor.extract_highlights(
            transcript=transcript_text,
            max_highlights=int(max_highlights),
            min_confidence=float(min_confidence)
        )
        
        # Error checking
        if 'error' in results['metadata']:
            error_msg = f"{results['metadata']['error']}"
            return error_msg, "", pd.DataFrame()
        
        # Result formatting
        display_text = format_results_for_display(results)
        highlights_table = create_highlights_table(results)
        json_output = json.dumps(results, indent=2, ensure_ascii=False)
        
        return display_text, json_output, highlights_table
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Processing error: {e}")
        return error_msg, "", pd.DataFrame()

def format_results_for_display(results):
    """Optimized formatting for display"""
    if not results['highlights']:
        return "No highlights found. Try lowering the confidence threshold or check transcript quality."
    
    lines = []
    meta = results['metadata']
    
    # Header
    lines.extend([
        "NLP EXTRACTION RESULTS",
        "=" * 60,
        f"📊 Sentences analyzed: {meta['total_sentences']}",
        f"🧠 Version: {meta.get('version', '1.0')}",
        f"😊 Average sentiment: {meta['avg_sentiment']:.3f}",
        f"🌍 Language: {meta['language']}"
    ])
    
    if meta.get('top_keywords'):
        keywords = ', '.join(meta['top_keywords'][:5])
        lines.append(f"🏷️ Keywords: {keywords}")
    
    # Highlights
    highlights = results['highlights']
    lines.extend([
        f"\nTOP {len(highlights)} HIGHLIGHTS:",
        "-" * 60
    ])
    
    for i, h in enumerate(highlights, 1):
        confidence_pct = h['confidence'] * 100
        confidence_bar = "█" * int(confidence_pct / 10)
        
        lines.extend([
            f"\nHIGHLIGHT #{i}",
            f"📈 Confidence: {confidence_pct:.0f}% {confidence_bar}",
            f"⏱️ Time: {h['start_time']}s - {h['end_time']}s"
        ])
        
        # Text with intelligent limit
        text = h['text']
        if len(text) > 200:
            # Cut at last complete word
            text = text[:197] + "..."
        lines.append(f'💬 "{text}"')
        
        # Metrics
        f = h['features']
        lines.extend([
            f"📊 Sentiment: {f['sentiment']:.2f} | Entities: {f['entities']} | " +
            f"Keywords: {f['keywords']} | TF-IDF: {f['tfidf_score']:.3f}",
            "-" * 40
        ])
    
    # Analytics
    if 'sentiment_distribution' in results['analytics']:
        sent = results['analytics']['sentiment_distribution']
        lines.extend([
            f"\n📊 GLOBAL SENTIMENT ANALYSIS:",
            f"😊 Positive: {sent['positive']:.1%} | " +
            f"😐 Neutral: {sent['neutral']:.1%} | " +
            f"😔 Negative: {sent['negative']:.1%}"
        ])
    
    return "\n".join(lines)

def create_highlights_table(results):
    """Table for Gradio interface"""
    if not results['highlights']:
        return pd.DataFrame()
    
    rows = []
    for i, h in enumerate(results['highlights'], 1):
        f = h['features']
        text = h['text'][:80] + "..." if len(h['text']) > 80 else h['text']
        
        rows.append({
            'Rang': f"#{i}",
            'Confiance': f"{h['confidence']:.0%}",
            'Temps': f"{h['start_time']}s",
            'Extrait': text,
            'Sentiment': f"{f['sentiment']:.2f}",
            'Entités': f['entities'],
            'Mots-clés': f['keywords']
        })
    
    return pd.DataFrame(rows)

def get_example_transcript(example_name):
    """Optimized examples"""
    examples = {
        "Podcast IA": """
        Bienvenue dans ce podcast dédié à l'intelligence artificielle. Aujourd'hui, nous explorons les avancées révolutionnaires qui transforment notre monde. L'IA n'est plus de la science-fiction, c'est une réalité qui impacte déjà tous les secteurs.
        
        Premier point crucial à retenir : les modèles de langage comme GPT ont changé la donne. Ils permettent désormais des interactions naturelles entre humains et machines. C'est un tournant historique dans l'histoire de l'informatique.
        
        Mon conseil pratique pour les entrepreneurs ? Intégrez l'IA dès maintenant dans vos processus. 80% des entreprises qui tardent à adopter ces technologies risquent d'être dépassées. Ne faites pas cette erreur !
        
        Attention cependant aux enjeux éthiques. L'IA soulève des questions fondamentales sur la vie privée, l'emploi et la prise de décision automatisée. Il est essentiel d'encadrer son développement.
        
        En conclusion, nous vivons une révolution technologique majeure. L'intelligence artificielle va redéfinir notre façon de travailler, d'apprendre et d'interagir. Préparez-vous à ce changement !
        """,
        
        "Interview Business": """
        Merci de m'recevoir pour partager mon parcours d'entrepreneur. Créer une startup, c'est avant tout une aventure humaine extraordinaire remplie de défis et d'apprentissages.
        
        La leçon la plus importante ? Il faut absolument valider son marché avant tout. 70% des startups échouent parce qu'elles créent des produits que personne ne veut ! C'est une erreur fatale mais évitable.
        
        Mon secret pour le financement : construisez d'abord une traction solide. Les investisseurs financent des résultats, pas des idées. Montrez-leur des métriques concrètes et une croissance régulière.
        
        Attention aux erreurs classiques ! Recruter trop vite, négliger la culture d'entreprise, ou vouloir tout faire soi-même. L'humilité et l'apprentissage constant sont cruciaux dans l'entrepreneuriat.
        
        En résumé, l'entrepreneuriat demande passion, résilience et pragmatisme. Si vous avez une vision claire et l'envie d'impact, lancez-vous ! Le monde a besoin d'innovation.
        """,
        
        "Cours Technique": """
        Aujourd'hui nous abordons les fondamentaux des architectures microservices, un sujet essentiel pour tout développeur moderne. Cette approche révolutionne la conception d'applications.
        
        Première règle fondamentale : un microservice doit avoir une responsabilité unique et bien définie. C'est le principe de responsabilité unique appliqué à l'architecture. Chaque service gère un domaine métier spécifique.
        
        Point crucial à retenir : la communication entre services. Privilégiez les API REST et les messages asynchrones. Évitez les couplages forts qui détruisent les bénéfices de l'architecture.
        
        Attention aux pièges ! La complexité opérationnelle augmente exponentiellement. Il faut maîtriser Docker, Kubernetes, la surveillance et le déplajement automatisé. Ne sous-estimez jamais cet aspect.
        
        En conclusion, les microservices offrent scalabilité et flexibilité, mais exigent maturité technique et organisationnelle. Commencez petit et évoluez progressivement vers cette architecture.
        """,
        
        "🌟 AI Enthusiasm": """
        Welcome to this incredible journey into the amazing world of artificial intelligence! Today we're exploring the most exciting and revolutionary breakthroughs that are absolutely transforming our digital landscape. This is truly a golden age of innovation!
        
        The most wonderful thing about modern AI is how it's making technology accessible to everyone. These incredible tools are opening up endless possibilities for creativity and problem-solving. It's absolutely mind-blowing how much progress we've made in just a few years!
        
        Here's the fantastic news: machine learning is now easier than ever to implement! The beautiful thing is that you don't need to be a genius mathematician anymore. These amazing libraries and frameworks handle all the complex calculations for you, making AI development incredibly accessible and enjoyable.
        
        The best part? The results are absolutely spectacular! We're seeing accuracy improvements that were unimaginable just a decade ago. It's like watching magic happen in real-time - the way these systems can understand, learn, and adapt is truly extraordinary.
        
        What makes this even more exciting is the incredible community behind it all. Developers from around the world are sharing their knowledge, creating amazing open-source projects, and helping each other succeed. The collaboration and innovation happening right now is absolutely inspiring!
        
        And here's the most beautiful thing: this technology is helping solve real-world problems! From healthcare breakthroughs to environmental protection, AI is making our world a better place. Every day brings new discoveries that fill us with hope and excitement for the future.
        
        The possibilities are absolutely limitless! Whether you're building the next breakthrough app, creating art with AI, or solving complex scientific problems, the tools are there and they're getting better every single day. This is truly the most exciting time to be in technology!
        
        Remember, the key to success is staying curious and never stopping learning. Every challenge is an opportunity for growth, and every failure is a stepping stone to something amazing. The future is bright, and it's happening right now!
        """
    }
    return examples.get(example_name, "")

# Gradio Interface
def create_interface():
    """Gradio interface optimized for production"""
    
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .gr-button {
        border-radius: 8px !important;
    }
    .gr-panel {
        border-radius: 12px !important;
    }
    """
    
    with gr.Blocks(
        title="Extracteur NLP de Highlights",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);">
            <h1 style="margin: 0; font-size: 2.8em; font-weight: 700;">Extracteur NLP de Highlights</h1>
            <p style="margin: 15px 0 0 0; font-size: 1.3em; opacity: 0.95;">Intelligence Artificielle pour l'identification automatique de moments clés dans vos podcasts</p>
            <p style="margin: 8px 0 0 0; font-size: 0.95em; opacity: 0.8;">Propulsé par NLTK, scikit-learn et spaCy • Version Production</p>
        </div>
        """)
        
        with gr.Row():
            # Parameters
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #667eea;'>⚙️ Configuration</h3>")
                
                language = gr.Dropdown(
                    choices=[("🇫🇷 Français", "fr"), ("🇬🇧 English", "en")],
                    value="fr",
                    label="Langue du transcript",
                    info="Optimise l'analyse selon la langue"
                )
                
                max_highlights = gr.Slider(
                    minimum=1, maximum=15, value=6, step=1,
                    label="Nombre de highlights",
                    info="Plus = plus d'extraits"
                )
                
                min_confidence = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                    label="Seuil de qualité",
                    info="Plus bas = plus de résultats"
                )
                
                gr.HTML("<h4 style='color: #764ba2; margin-top: 25px;'>Examples</h4>")
                
                example_buttons = []
                for example in ["Podcast IA", "Interview Business", "Cours Technique", "🌟 AI Enthusiasm"]:
                    btn = gr.Button(example, size="sm", variant="secondary")
                    example_buttons.append((btn, example))
            
            # Main area
            with gr.Column(scale=2):
                gr.HTML("<h3 style='color: #667eea;'>Transcript à analyser</h3>")
                
                transcript_input = gr.Textbox(
                    placeholder="Collez ici le transcript de votre podcast, interview, cours ou présentation...\n\n💡 Conseil: Plus le contenu est riche et structuré, meilleurs seront les résultats!\n\n📊 Minimum recommandé: 300 mots pour des résultats optimaux.",
                    lines=15,
                    label="Contenu à analyser",
                    info="Transcript, interview, cours, présentation..."
                )
                
                with gr.Row():
                    extract_btn = gr.Button(
                        "Analyser et Extraire", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button("Effacer", variant="secondary", scale=1)
        
        # Results
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3 style='color: #667eea;'>Highlights Extraits</h3>")
                results_display = gr.Textbox(
                    label="Résultats de l'analyse",
                    lines=25,
                    max_lines=40,
                    container=True
                )
        
        # Table and JSON
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3 style='color: #764ba2;'>Tableau Détaillé</h3>")
                highlights_table = gr.DataFrame(
                    label="Analyse des highlights",
                    interactive=False,
                    wrap=True
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #764ba2;'>Export JSON</h3>")
                json_output = gr.Code(
                    label="Données structurées",
                    language="json",
                    lines=15
                )
        
        # Event handlers
        extract_btn.click(
            fn=process_transcript,
            inputs=[transcript_input, language, max_highlights, min_confidence],
            outputs=[results_display, json_output, highlights_table]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", pd.DataFrame()),
            outputs=[transcript_input, results_display, highlights_table]
            )
        
        # Example buttons
        for btn, example_name in example_buttons:
            btn.click(
                fn=lambda name=example_name: get_example_transcript(name),
                outputs=[transcript_input]
            )
    
    return interface

# Main execution block for deployment
if __name__ == "__main__":
    # Create the interface
    interface = create_interface()
    
    # Configure for production deployment
    interface.launch(
        server_name="0.0.0.0",  # Required for Render
        server_port=int(os.environ.get("PORT", 7860)),  # Use Render's PORT
        share=False,  # Disable sharing for production
        debug=False,  # Disable debug mode for production
        show_error=True,  # Show errors in production
        quiet=False  # Show startup messages
    )