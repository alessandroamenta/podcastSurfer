from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sklearn.cluster import KMeans
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import numpy as np
from langchain.chains import RetrievalQA
import logging
from langchain.prompts import PromptTemplate


logging.basicConfig(level=logging.INFO) 


# Initialize global variables
global_faiss_db = None
global_documents = None

# Function to reset global variables
def reset_globals():
    global global_faiss_db, global_documents
    global_faiss_db = None
    global_documents = None

def init_faiss_db(openai_api_key):
    global global_faiss_db, global_documents
    if global_faiss_db is None and global_documents is not None:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        global_faiss_db = FAISS.from_documents(global_documents, embeddings)

def process_and_cluster_captions(captions, openai_api_key, num_clusters=17):
    global global_documents
    logging.info("Processing and clustering captions")
    
    # Log the first 500 characters of the captions to check their format
    logging.info(f"Captions received (first 500 characters): {captions[0].page_content[:500]}")
    caption_content = captions[0].page_content

    # Ensure captions is a string before processing
    if not isinstance(caption_content, str):
        logging.error("Captions are not in the expected string format")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.create_documents([caption_content])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in split_docs])
    
    # Log the first few hundred characters of the embeddings
    logging.info(f"Embeddings generated (preview): {str(vectors)[:5000]}")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = [np.argmin(np.linalg.norm(vectors - center, axis=1)) for center in kmeans.cluster_centers_]
    representative_docs = [split_docs[i] for i in closest_indices]
    
    # Log representative_docs with a limit on the output length
    logging.info(f"Clustering completed. Representative documents (preview): {str(representative_docs)[:500]}")
    global_documents = split_docs
    init_faiss_db(openai_api_key) 
    return representative_docs


def generate_summary(representative_docs, openai_api_key):
    logging.info("Generating summary")
    llm4 = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.2, openai_api_key=openai_api_key)

    # Concatenate texts for summary
    summary_text = "\n".join([doc.page_content for doc in representative_docs])

    summary_prompt_template = PromptTemplate(
        template=(
            "Create a concise summary of a podcast conversation based on the text provided below. The text consists of selected, representative sections from different parts of the conversation. "
            "Your task is to synthesize these sections into a single cohesive and concise summary. Focus on the overarching themes and main points discussed throughout the podcast. "
            "The summary should give a clear and complete understanding of the conversation's key topics and insights, while omitting any extraneous details. It should be engaging and easy to read, ideally in one or two paragraphs. Keep it short where possible"
            "\n\nSelected Podcast Sections:\n{text}\n\nSummary:"
        ),
        input_variables=["text"]
    )
    # Load summarizer chain
    summarize_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=summary_prompt_template)

    # Run the summarizer chain
    summary = summarize_chain.run([Document(page_content=summary_text)])

    logging.info("Summary generation completed")
    return summary


def answer_question(question, openai_api_key):
    llm4 = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.2, openai_api_key=openai_api_key, max_tokens=350)
    global global_faiss_db
    if global_faiss_db is None:
        init_faiss_db(openai_api_key) 
    logging.info(f"Answering question: {question}")
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm4, 
        chain_type="stuff", 
        retriever=global_faiss_db.as_retriever(search_type="similarity", search_kwargs={"k":8}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    response = qa_chain({"query": question})
    logging.info(f"this is the result: {response}")
    output = response['result']
    
    return output