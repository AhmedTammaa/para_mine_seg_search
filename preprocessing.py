import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from unicodedata import normalize
import spacy


def text_normalization(text):
    """ 
    Perform text normalization on a paragraph of text.

    This involves:
    - Lowercasing 
    - Fixing whitespace issues
    - Removing punctuation
    - Expanding contractions
    - Removing accented characters
    - Lemmatizing text  
    - Removing stop words

    It returns the normalized text as a string.
    """

    # Lowercase
    text = text.lower()

    # Fix whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Expand contractions
    contractions = {"ain't": "am not", "aren't": "are not"}
    text = text.replace("n't", " not")
    text = re.sub('|'.join(contractions.keys()),
                  lambda x: contractions[x.group()], text)

    # Remove accented characters
    text = (normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore'))

    # Get root form of words (lemmatize)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split()
                    if word not in stop_words])

    return text


def postprocess_titles(title):
    nltk.download('punkt')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(title)

    # Take first sentence
    full_sent = list(doc.sents)[0].text

    # Remove repeating words
    tokens = [token.text for token in doc if not token.is_stop]
    unique_tokens = dict((x, True) for x in tokens).keys()

    cleaned_title = " ".join(unique_tokens)

    # If empty, set default
    if not cleaned_title.strip():
        cleaned_title = "Untitled"

    return cleaned_title
