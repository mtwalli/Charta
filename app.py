import streamlit as st
import pymupdf
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import pandas as pd

def get_pdf_text(files):
    """
    Extracts the text content from a list of PDF files.

    Args:
        files (list): A list of file objects representing PDF files.

    Returns:
        str: The concatenated text content from all the pages of the PDF files.
    """
    print(files)
    raw_text = ""
    for file in files:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        for page in doc:
            raw_text += page.get_text()
    return raw_text

def get_text_chunks(raw_text):
    """
    Splits the given raw text into chunks of specified size.

    Args:
        raw_text (str): The raw text to be split into chunks.

    Returns:
        list: A list of text chunks.

    """
    character_text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = character_text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    # embeddings = HuggingFaceEmbeddings(model_name="nvidia/NV-Embed-v2")
    return FAISS.from_texts(text_chunks, embeddings)

def get_chain(vectorstore):
    """
    Returns a ConversationalRetrievalChain object initialized with the given parameters.

    Parameters:
    - vectorstore: The vectorstore used as a retriever for the ConversationalRetrievalChain.

    Returns:
    - chain: A ConversationalRetrievalChain object.

    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

def handle_question(user_question):
    """
    Handles a user question and generates a response.

    Parameters:
    user_question (str): The question asked by the user.

    Returns:
    None
    """

    response = st.session_state.chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    chat_container = st.container()
    with chat_container:
        for i,message in enumerate(st.session_state.chat_history):
            role = None
            if type(message).__name__ == 'HumanMessage': 
                role = "human" 
            else: 
                role = "assistant"

            with st.chat_message(role):
                st.markdown(message.content)

def get_data_frame(vector_store):
    """
    Converts a Faiss vector store into a pandas DataFrame.

    Args:
        vector_store (faiss.vector_store): The Faiss vector store.

    Returns:
        pandas.DataFrame: The DataFrame containing the vectors from the vector store.
    """
    faiss_index = vector_store.index
    num_vectors = faiss_index.ntotal
    vectors = [faiss_index.reconstruct(i) for i in range(num_vectors)]
    df = pd.DataFrame(vectors)
    return df


def main():
    load_dotenv()
    st.set_page_config(page_title="DocuAI", page_icon=":sparkles:",layout="wide")
    
    if "chain" not in st.session_state: 
        st.session_state.chain = None

    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = None

    st.header("DocuAI :sparkles:")
    st.markdown("DocuAI is a conversational AI that can help you understand the content of your documents. Upload your PDFs and ask questions about the content.")
    
    
    left, right = st.columns(2, gap='large')

    with left:
        st.subheader("Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.text_area("Raw Text", raw_text)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write("Text Chunks:")
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vector_store(text_chunks)
                st.write("Vector Store:")
                st.write(get_data_frame(vectorstore))

                # create chain
                st.session_state.chain = get_chain(vectorstore)

    with right:
        if st.session_state.chain:
            st.subheader("Chat with DocuAI")
            user_question = st.text_input("Ask a question")
            if user_question:
                with st.spinner("Thinking..."):
                    handle_question(user_question) 
             

if __name__ == "__main__":
    main()