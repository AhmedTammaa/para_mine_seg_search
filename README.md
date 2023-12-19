# Paragraph Retrieval and Headline Generation
This repository contains a document retrieval system that extracts relevant paragraphs based on a given topic from uploaded documents and automatically generates a headline for each retrieved paragraph.

# Table of Contents
* Overview
* Usage
* Pipeline Architecture
* Document Preprocessing
* Paragraph Retrieval
* Headline Generation
* Search
* Installation
* Contributing
* Contact

# Overview

## The system allows users to:

* Upload documents in various formats (PDF, DOC(X), TXT)
* Automatically divide text into paragraphs by semantic similarity
* Specify a search topic/query
* Retrieves relevant paragraphs based on semantic search
* Automatically generates engaging headlines for each paragraph
* Allows searching retrieved paragraphs
* It focuses on paragraph-level retrieval as opposed to just matching keywords or whole documents. This provides readers with more focused and contextual information.

## The project objectives were to:

* Develop paragraph-based retrieval using transformer models to match paragraph embeddings
* Implement an abstractive summarization model to generate headers
* Build document ingestion and processing pipelines using SKLearn
* Create scripts to simulate document uploads and management
* Evaluate retrieval precision, headline quality, and system efficiency

# Usage
The system is comprised of Python scripts and Jupyter notebooks.

To use the paragraph retrieval and headline generation scripts:

Ensure all required libraries are installed
Place documents to process in input_documents/ folder
Deployed Link Soon.
<!---->
Run python process_documents.py to trigger the pipeline
<!---->
Retrieved paragraphs and headlines will be saved to output/
To search paragraphs:

Run app.py

Enter a search query in the UI
Results will display paragraph matches and similarity score
Pipeline Architecture
The system follows a pipeline architecture:

Pipeline Architecture
(soon)

Document Preprocessing

Uploads are normalized via steps:

# Text extraction (PDF, DOCX)
Tokenization
Paragraph segmentation
Paragraph Retrieval
all-mpnet-base-v2 model encodes paragraphs and topic search vectors to vectors. DBSCAN clustering extracts related paragraph clusters.

# Headline Generation
T5 Headline Generator creates abstractive headlines for retrieved paragraphs.

# Search
all-mpnet-base-v2 encodes search queries and paragraphs. Matches are ranked by cosine similarity.

# Installation


pip install -r requirements.txt
Requirements are stored in requirements.txt and include:

Transformers
spaCy
scikit-learn
Streamlit

Contributing
Contributions to improve the paragraph segmentation, search accuracy, headline quality and processing speeds are welcome. Please open an issue first to discuss the proposal before submitting PR.

Contact
For any queries, please reach out to ahmedtammaa101@gmail.com
