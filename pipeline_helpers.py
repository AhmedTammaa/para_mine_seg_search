
from sklearn.base import BaseEstimator, TransformerMixin

import utils
from predictors import predict_title, segment_documents_into_paragraphs
from preprocessing import postprocess_titles
import nltk
from config import UPLOAD_FOLDER
import streamlit as st

class ExtractAndPredict(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # nltk.download('punkt')
        st.write(type(X))
        text = utils.process_folder(X)
        # Concatenate text into one long string
        text = "".join(text.values())
        # paragraphs = predict_paragraphs(text, self.dbscan, self.vectorizer)
        # print(type(text))

        paragraphs = segment_documents_into_paragraphs(
            [text], eps=0.5, min_samples=2, min_paragraph_size=3, max_paragraph_size=10)
        return paragraphs


class CleanParagraphs(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, paragraphs):
        # clean = [text_normalization(p) for p in paragraphs]
        return paragraphs


class TitleGenerator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.tokenizer, self.model = utils.load_title_generator()
        self.orig_paragraphs = []

    def fit(self, X, y=None):
        self.orig_paragraphs = X
        return self

    def transform(self, clean_paragraphs):
        results = []
        for idx, para in enumerate(clean_paragraphs):
            title = predict_title(para, self.tokenizer, self.model)
            title = postprocess_titles(title)
            # orig = self.orig_paragraphs[idx]
            results.append((idx, para, title))

        return results
