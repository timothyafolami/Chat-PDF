import streamlit as st
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai import genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfFileReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()  
    return text

def get_text_chunks(text, chunk_size=10000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding=001")
    vector_store = FAISS(chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in the provided context, just say "answer is not available in the context", don't provide wrong answer.
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:

    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model=model, chain_type="studd", prompt=prompt_template)
    return chain


def use_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding=001")

    new_db = FAISS.load_local("faiss_index", embeddings)

    docs = new_db.search(user_question)

    chain =  get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Enter your question for the PDF files: ")

    if user_question:
        use_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
    