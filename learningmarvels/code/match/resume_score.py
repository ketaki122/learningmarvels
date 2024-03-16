import re
import os
import string
import nltk
import torch
import PyPDF2
import json
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove emails
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def stem_words(tokens):
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens

def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    return lang

def preprocess_text(text):
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize text
    tokens = tokenize_text(cleaned_text)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Lemmatize words
    tokens = lemmatize_words(tokens)
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def preprocess_text_with_ner(text):
    # Process text with spaCy NER pipeline
    doc = nlp(text)
    
    # Initialize list to store tokens
    processed_tokens = []
    
    # Extract named entities and their labels
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Add named entities to tokens list
    for token, label in entities:
        processed_tokens.append(token.lower())  # Convert to lowercase for consistency
    
    # Tokenize remaining text and add to tokens list
    for token in doc:
        if token.ent_type_ == '':
            processed_tokens.append(token.lower())  # Convert to lowercase for consistency
    
    # Join tokens back into text
    processed_text = ' '.join(processed_tokens)
    
    return processed_text


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def calculate_similarity(jd_embedding, resume_embedding):
    return cosine_similarity(jd_embedding, resume_embedding).item()

def match_resumes_to_jd(jd_text, resume_folder, threshold, top_n):
    confidence_score = 0.0

    if not jd_text or not resume_folder:
        return json.dumps({"error": "Empty input provided."}, indent=4)

    # Tokenize and encode JD
    jd_tokens = tokenizer.encode(jd_text, add_special_tokens=True, truncation=True, max_length=512)
    jd_tensor = torch.tensor(jd_tokens).unsqueeze(0)
    
    # Get BERT embeddings for JD
    with torch.no_grad():
        jd_outputs = model(jd_tensor)
        jd_embedding = jd_outputs[0][:, 0, :].numpy()  # Take the [CLS] token embedding
    
    
    # Iterate through resume files in the folder
    resume_scores = []
    print(resume_folder)
    for filename in os.listdir(resume_folder):
        if filename.endswith('.pdf'):  # Assuming resumes are PDF files
            # Read and preprocess resume text
            resume_text = extract_text_from_pdf(os.path.join(resume_folder, filename))
            processed_resume_text = preprocess_text(resume_text)
            
            
            # Tokenize and encode resume
            resume_tokens = tokenizer.encode(processed_resume_text, add_special_tokens=True, truncation=True, max_length=512)
            resume_tensor = torch.tensor(resume_tokens).unsqueeze(0)
            
            # Get BERT embeddings for resume
            with torch.no_grad():
                resume_outputs = model(resume_tensor)
                resume_embedding = resume_outputs[0][:, 0, :].numpy()  # Take the [CLS] token embedding
            
            # # Debugging: Print resume embedding
            # print("Resume Embedding Shape:", resume_embedding.shape)
            # print("Resume Embedding:", resume_embedding)
            
            # Calculate similarity score
            similarity_score = calculate_similarity(jd_embedding, resume_embedding)
            similarity_score = round(similarity_score, 2)
            
            # Debugging: Print similarity score
            #print("Similarity Score:", similarity_score)
            
            # If similarity score exceeds threshold, add to results
            if  threshold <= similarity_score:
                resume_scores.append((filename, similarity_score))
                
    # Sort resumes based on similarity scores
    sorted_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)
    
    # Return top N results if available, otherwise return all results
    if len(sorted_resumes) >= top_n:
        top_results = sorted_resumes[:top_n]
    else:
        top_results = sorted_resumes

        # Calculate confidence score based on similarity scores of top N results
    top_results_scores = [score for _, score in top_results]
    if top_results_scores:
        total_similarity_score = sum(top_results_scores)
        average_similarity_score = total_similarity_score / len(top_results_scores)
        confidence_score = round(average_similarity_score, 2)
    else:
        confidence_score = 0.0

        # Prepare results in JSON format
    results_json = {
        "count": len(top_results),
        "metadata": {
            "confidenceScore": confidence_score
        },
        "results": []
    }
    
    for idx, (filename, score) in enumerate(top_results, start=1):
        result_entry = {
            "id": idx,
            "score": score,
            "path": filename
        }
        results_json["results"].append(result_entry)    
    return json.dumps(results_json, indent=4)


def find_top_matching_resumes(jd_text, resume_folder, top_n=3):
    # Extract text from job description and resumes
    jd_text = preprocess_text(jd_text)
    resume_texts = []
    for filename in os.listdir(resume_folder):
        if filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(os.path.join(resume_folder, filename))
            resume_texts.append(preprocess_text(resume_text))

    # Vectorize job description and resumes using TF-IDF
    vectorizer = TfidfVectorizer()
    jd_vector = vectorizer.fit_transform([jd_text])
    resume_vectors = vectorizer.transform(resume_texts)

    # Calculate cosine similarity between job description and resumes
    similarity_scores = cosine_similarity(jd_vector, resume_vectors)[0]

    # Get indices of top matching resumes
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    #top_indices = round(top_indices,2)
    # Get paths of top matching resumes
    top_resumes = [os.listdir(resume_folder)[i] for i in top_indices]

    return top_resumes, similarity_scores[top_indices]



# Example usage
#jd = "java developer with 5yrs"
#resume_folder = "../resume-data"
#r"C:\\TESTLAB\\learningmarvels\\code\\resume-data"
#top_n = 4
#threshold= 0.5


#recommended_resumes = match_resumes_to_jd(jd, resume_folder, threshold, top_n)
#print(recommended_resumes)


# topresumes,similarity_scores  = find_top_matching_resumes(jd, resume_folder, top_n=3)
# for resume, score in zip(topresumes, similarity_scores):
#     print("Resume:", resume)
#     print("Similarity Score:", score)
