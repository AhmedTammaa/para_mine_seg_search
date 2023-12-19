import pandas as pd
import joblib
import os
from docx import Document
import PyPDF2

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import dominate
from dominate.tags import div, p, style, h1


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
        "fabiochiu/t5-base-medium-title-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "fabiochiu/t5-base-medium-title-generation")
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
    dbscan = joblib.load(
        "https://drive.google.com/file/d/13XPrwL9hBnBGlCfH-oN0CAF8QVlsJeRx/view?usp=drive_link")
    vectorizer = joblib.load(
        "https://drive.google.com/file/d/1-6VlNND0w77IKx9XixFgKS1Z7LhZ_FTu/view?usp=drive_link")
    return dbscan, vectorizer


def extract_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".txt":
        try:
            # Try reading the file with UTF-8
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()

    elif file_extension == ".pdf":
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_number in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_number].extract_text()
            return text
    elif file_extension == ".docx":
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    else:
        print(f"Unsupported file format: {file_extension}")
        return None


def process_folder(folder_path):
    extracted_data = dict()

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            extracted_text = extract_text_from_file(file_path)
            if extracted_text is not None:
                extracted_data[file_path] = extracted_text
    return extracted_data
