import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data on import
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class NLPService:
    """
    Service for preprocessing text data using NLTK.
    Handles tokenization, stop word removal, and lemmatization.
    """

    def __init__(self, language='portuguese'):
        """
        Initialize the NLP service.
        
        Args:
            language: Language for stopwords (default: 'portuguese')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words(language))
        except OSError:
            logger.warning(f"Stopwords for language '{language}' not available, using English")
            self.stop_words = set(stopwords.words('english'))

    def preprocessar_texto(self, text):
        """
        Preprocess the input text.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove special characters and punctuation
        5. Tokenize
        6. Remove stop words
        7. Lemmatize
        
        Args:
            text: Input text to preprocess
            
        Returns:
            str: Preprocessed text (cleaned and lemmatized)
            list: List of tokens
        """
        if not text:
            return "", []

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters, keeping only alphanumeric (including accented letters) and spaces
        # Use Unicode-aware \w so accents are preserved
        text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            logger.warning("Tokenization failed, using split instead")
            tokens = text.split()

        # Remove stop words and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:  # Filter short words
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)

        # Reconstruct text from processed tokens
        processed_text = ' '.join(processed_tokens)

        return processed_text, processed_tokens

    def extrair_features(self, text):
        """
        Extract useful features from text.
        
        Args:
            text: Input text
            
        Returns:
            dict: Dictionary with extracted features
        """
        processed_text, tokens = self.preprocessar_texto(text)

        return {
            'original_length': len(text),
            'processed_length': len(processed_text),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'processed_text': processed_text,
            'tokens': tokens,
        }


# Create a singleton instance (Portuguese for BR email processing)
nlp_service = NLPService(language='portuguese')
