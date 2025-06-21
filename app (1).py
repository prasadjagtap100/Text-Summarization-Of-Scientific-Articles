import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences
splitter = SentenceSplitter(language='en')
import re

s_model_name = "allenai/scibert_scivocab_uncased"
s_tokenizer = AutoTokenizer.from_pretrained(s_model_name)
s_model = AutoModel.from_pretrained(s_model_name)
s_model.eval()  # set model to evaluation mode 


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# Load pre-trained BART model and tokenizer
B_model_name = "facebook/bart-large-cnn"
B_tokenizer = BartTokenizer.from_pretrained(B_model_name)
B_model = BartForConditionalGeneration.from_pretrained(B_model_name)

@st.cache_resource
def scibert_generated_summary(input_text):

    #scibert summarization
    # Step 1: Sentence Segmentation
    sentences = sent_tokenize(input_text)
     
    # Step 2: Encode each sentence using SciBERT
    def encode_text(text):
        """Encodes text and returns the [CLS] token embedding."""
        inputs = s_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
           outputs = s_model(**inputs)
        # Using the [CLS] token embedding as a representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_embedding
    
    # Encode all sentences
    sentence_embeddings = [encode_text(sent) for sent in sentences]

    # Step 3: Compute a simple document embedding by averaging sentence embeddings
    document_embedding = np.mean(sentence_embeddings, axis=0, keepdims=True)

    # Step 4: Compute cosine similarity of each sentence with the document embedding
    # Reshape each sentence embedding to 2D for cosine_similarity computation
    similarity_scores = [
        cosine_similarity(embedding.reshape(1, -1), document_embedding)[0][0]
        for embedding in sentence_embeddings
    ]
    
    # For example, select the top 3 sentences.
    k = 10
    
    top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

    top_k_indices_sorted = sorted(top_k_indices)
    scibert_summary = "\n".join([sentences[i] for i in top_k_indices_sorted])

    return scibert_summary


@st.cache_resource
def graph_generatred_summary(input_text,num_sentences=10):

    sentences = nltk.sent_tokenize(input_text)
    
    # Step 3: Compute TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(vectorizer)
    
    # Step 4: Build graph and apply PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # Step 5: Rank sentences and pick top N
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    graph_summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    
    return graph_summary


@st.cache_resource
def bart_summary(sentence3, max_length=400, min_length=50):
    """Generate an abstractive summary using BART"""
    # Tokenize input text
    inputs = B_tokenizer(sentence3, return_tensors="pt", truncation=True, max_length=1024)

    # Generate summary
    summary_ids = B_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.5,
        num_beams=6,
        no_repeat_ngram_size=3,  
        early_stopping=True,
    )

    # Decode and return the summary
    return B_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@st.cache_resource
def get_response(input_text,num_return_sequences):
      batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
      translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
      tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
      return tgt_text



   

# # Streamlit App
st.title("Scientific Text Summarization ")

# Tab 1: Summarization
st.header("Generate your Summary")
text = st.text_area("Enter Text for Summarization", height=200)

# input_text = preprocessing(text)
input_text = re.sub(r'[(){}\[\]]', '',text)



    
  
if st.button("Generate Summary"):
    sentence1 = scibert_generated_summary(input_text)
    sentence2 = graph_generatred_summary(input_text)
    sentence3 = sentence1 + " "+ sentence2
    summary = bart_summary(sentence3)

    sentence_list = splitter.split(summary)
    # Do a for loop to iterate through the list of sentences and paraphrase each sentence in the iteration
    paraphrase = []
    for i in sentence_list:
     a = get_response(i,1)
     paraphrase.append(a)

    paraphrase2 = [' '.join(x) for x in paraphrase]

    # Combines the above list into a paragraph
    paraphrase3 = [' '.join(x for x in paraphrase2) ]
    paratext = str(paraphrase3).strip('[]').strip("'")


    st.success(f"{paratext}")

    # st.success(f"{summary}")





