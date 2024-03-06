import os
import streamlit as st
import openai
from pinecone import Pinecone


openapikey = st.secrets["OPENAPIKEY"]
print(openapikey)
# Initialize Pinecone client
pc = Pinecone(api_key="10c85a4c-db09-45a6-93dd-2e302ba1b7cb")
index = pc.Index("canopy--document-uploader")

# Set your OpenAI API key here
#openai.api_key = "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"
openai.api_key = openapikey
def generate_embeddings(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    embedding_vector = response['data'][0]['embedding']
    return embedding_vector

def get_relevant_kb_entries(query):
    query_vector = generate_embeddings(query)
    response = index.query(vector=query_vector, top_k=3, namespace="", include_values=True, include_metadata=True)
    results = []
    for match in response.matches:
        meta = match.metadata if 'metadata' in match else None
        if meta and 'text' in meta:
            results.append(meta['text'])
    
    return results if results else ["No relevant entries found or missing metadata."]

def get_chat_completion(user_query):
    relevant_kb_entries = get_relevant_kb_entries(user_query)
    context = "\n".join(relevant_kb_entries)
    full_context = f"The following information from the knowledge base is relevant:\n{context}\n\n{user_query}"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant which replies so kids can understand spirtal text"},
            {"role": "user", "content": full_context}
        ]
    )
    return response.choices[0].message['content'] if response.choices else "No response generated."

# Streamlit UI
st.title("Bhagwad Gita Query Assistant")

user_query = st.text_input("Enter your query:", "")
if user_query:
    with st.spinner('Getting response...'):
        response = get_chat_completion(user_query)
        st.write("You:", user_query)
        st.write("Bot:", response)


