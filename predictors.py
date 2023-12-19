from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy
from utils import load_models
from tqdm.notebook import tqdm
from evaluation import evaluate_clusters
from preprocessing import text_normalization
import nltk
from sentence_transformers import SentenceTransformer, util
import torch


def search_paragraphs(query, paragraphs):
    nlp = spacy.load("en_core_web_sm")

    # Load Sentence Transformer model
    model_name = "all-mpnet-base-v2"
    sentence_transformer = SentenceTransformer(model_name)

    # Extract content from the list of tuples
    paragraph_contents = [title + " " + content for idx,
                          content, title in paragraphs]

    query_embedding = sentence_transformer.encode([query])[0]

    # Calculate paragraph embeddings
    paragraph_embeddings = sentence_transformer.encode(paragraph_contents)

    # Calculate cosine similarity
    scores = util.pytorch_cos_sim(torch.tensor(
        [query_embedding]), torch.tensor(paragraph_embeddings))[0]
    scores = scores.cpu().numpy()

    # Get the indices of top results
    top_indices = scores.argsort()[::-1][:5]

    # Create a list of tuples with score, index, and content
    top_results = [(scores[idx], idx, paragraphs[idx][2],
                    paragraphs[idx][1]) for idx in top_indices]

    return top_results


def predict_title(text, tokenizer, model):

    inputs = tokenizer(["summarize: " + text],
                       truncation=True,
                       max_length=512,
                       return_tensors="pt")

    output = model.generate(**inputs,
                            num_beams=8,
                            max_length=100)

    title = tokenizer.decode(output[0], skip_special_tokens=True)

    sentences = nltk.sent_tokenize(title)
    if len(sentences) > 0:
        return sentences[0]

    else:
        # If no sentences detected
        # Return full string
        return title


def predict_paragraphs(new_text, vectorizer, dbscan_model):
    # Tokenize and preprocess sentences using spaCy
    nlp = spacy.load("en_core_web_sm")
    new_sentences = [sent.text for sent in nlp(new_text).sents]

    # TF-IDF Vectorization using the pre-trained vectorizer
    new_X = vectorizer.transform(new_sentences)

    # Calculate cosine similarity between new sentences and existing sentences
    similarity_matrix = cosine_similarity(new_X)

    # Predict clusters using the pre-trained DBSCAN model
    new_clusters = dbscan_model.fit_predict(similarity_matrix)

    # Map sentences to clusters
    new_sentence_clusters = {}
    for i, cluster_label in enumerate(new_clusters):
        if cluster_label not in new_sentence_clusters:
            new_sentence_clusters[cluster_label] = []
        new_sentence_clusters[cluster_label].append(new_sentences[i])

    # Extract paragraphs based on clusters
    new_paragraphs = [" ".join(cluster)
                      for cluster in new_sentence_clusters.values()]
    silhouette_avg = evaluate_clusters(new_X, new_clusters)
    print(f"Silhouette Score: {silhouette_avg}")
    return new_paragraphs


def paragraph_processing(extracted_text):
    dbscan, vectorizer = load_models()

    saved_paragraphs = dict()
    unnormalized_paragraph = dict()
    cnt = 0
    for i in tqdm(extracted_text):
        new_text = extracted_text[i]

        # print(new_text)
        if (len(new_text) < 100):
            print("Length should be longer than 100")
            continue
        # Use the predict function with the pre-trained vectorizer and DBSCAN model
        predicted_paragraphs = predict_paragraphs(new_text, vectorizer, dbscan)

        # Print the result
        for i, paragraph in enumerate(predicted_paragraphs, start=1):
            # print(f"Predicted Paragraph {i}:\n{paragraph}\n")
            unnormalized_paragraph[cnt] = paragraph.replace(
                '\n', '')  # used later for reterival
            saved_paragraphs[cnt] = text_normalization(paragraph)
            cnt = cnt + 1


def segment_documents_into_paragraphs(document_list, eps=0.5, min_samples=2, min_paragraph_size=3, max_paragraph_size=10):
    nltk.download('punkt')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paragraphs = []

    for document in document_list:
        # Split document into sentences
        sentences = nltk.sent_tokenize(document)

        # Generate embeddings for each sentence
        embeddings = model.encode(sentences, convert_to_tensor=True)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples,
                            metric='cosine').fit(embeddings.cpu())
        labels = clustering.labels_

        # Initial grouping of sentences into paragraphs
        temp_paragraphs = []
        paragraph = []
        for sentence, label in zip(sentences, labels):
            if label == -1 or len(paragraph) >= max_paragraph_size:
                if paragraph:
                    temp_paragraphs.append(paragraph)
                    paragraph = []
                temp_paragraphs.append([sentence])
            else:
                paragraph.append(sentence)

        if paragraph:
            temp_paragraphs.append(paragraph)

        # Post-processing: Adjusting paragraph sizes
        for paragraph in temp_paragraphs:
            if len(paragraph) < min_paragraph_size:
                if paragraphs:
                    paragraphs[-1].extend(paragraph)
                else:
                    paragraphs.append(paragraph)
            else:
                paragraphs.append(paragraph)

        # Splitting large paragraphs
        final_paragraphs = []
        for paragraph in paragraphs:
            if len(paragraph) > max_paragraph_size:
                for i in range(0, len(paragraph), max_paragraph_size):
                    final_paragraphs.append(
                        paragraph[i:i + max_paragraph_size])
            else:
                final_paragraphs.append(paragraph)

    # Convert list of sentences to text paragraphs
    paragraphs_text = [' '.join(p) for p in final_paragraphs]

    return paragraphs_text
