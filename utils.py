import pandas as pd
import joblib
import os
from docx import Document
import PyPDF2
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import dominate
from dominate.tags import div, p, style, h1
import gdown
import streamlit as st

def convert_to_html(results, file_path):
    doc = dominate.document(title='Processed Document')

    STYLE = """
    body {
        font-family: 'Georgia', serif;
        margin: 0;
        padding: 0;
        background: #fff;
        color: #333;
        line-height: 1.6;
    }
    .container {
        width: 80%;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    h1.title {
        text-align: center;
        font-size: 28px;
        margin-top: 50px;
        margin-bottom: 50px;
    }
    .subtitle {
        font-weight: bold;
        font-size: 20px;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .paragraph {
        font-size: 18px;
        margin-bottom: 20px;
        text-align: justify;
        text-indent: 40px;
    }
    """

    with doc.head:
        style(STYLE)

    with doc:
        with div(cls='container'):
            h1('Processed Document', cls='title')
            for idx, content, title in results:
                with div().add(p(f"{idx+1}. {title}", cls='subtitle')):
                    p(content, cls='paragraph')

    html_file = open(file_path, 'w')
    html_file.write(doc.render())
    return doc


def load_title_generator():
    tokenizer = AutoTokenizer.from_pretrained(
        "fabiochiu/t5-small-medium-title-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "fabiochiu/t5-small-medium-title-generation")
    return tokenizer, model


def save_paragraphs(unnormalized_paragraph, saved_paragraphs):
    df = pd.DataFrame.from_dict(
        saved_paragraphs, orient='index', columns=['para_content'])
    df.index.name = 'para_idx'
    df.reset_index(inplace=True)
    df.to_csv('saved_paragraphs.csv', header=True, index=False)
    df = pd.DataFrame.from_dict(
        unnormalized_paragraph, orient='index', columns=['para_content'])
    df.index.name = 'para_idx'
    df.reset_index(inplace=True)
    df.to_csv('original_paragraphs.csv', header=True, index=False)


def load_models():
    dbscan_link = "https://drive.google.com/file/d/13XPrwL9hBnBGlCfH-oN0CAF8QVlsJeRx/view?usp=sharing"
    vectorizer_link = "https://drive.google.com/file/d/1-6VlNND0w77IKx9XixFgKS1Z7LhZ_FTu/view?usp=sharing"
    dbscan_path = "dbscan.pkl"
    vectorizer_path = "vectorizer.pkl"

    gdown.download(vectorizer_link, vectorizer_path, quiet=False)
    gdown.download(dbscan_link, dbscan_path, quiet=False)
    dbscan = joblib.load(dbscan_path)

    vectorizer = joblib.load(vectorizer_path)
    return dbscan, vectorizer


def extract_text_from_file(uploaded_file):
    # Get the file extension from the uploaded file
    _, file_extension = os.path.splitext(uploaded_file.name.lower())

    if file_extension == ".txt":
        try:
            # Try reading the file with UTF-8
            return uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 fails, try reading with Latin-1
            uploaded_file.seek(0)
            return uploaded_file.read().decode("latin-1")

    elif file_extension == ".pdf":
        # Use PyPDF2 to extract text from PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_number in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_number].extract_text()
        return text
    elif file_extension == ".docx":
        # Use python-docx to extract text from DOCX
        doc = Document(BytesIO(uploaded_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None



def process_folder(folder_path):
    extracted_text = extract_text_from_file(folder_path)
    return extracted_text
