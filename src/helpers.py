import pymupdf
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
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
    raw_text = ""
    for file in files:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        for page in doc:
            raw_text += page.get_text()
    return raw_text

def get_text(files):
    """
    Extracts the text content from a list of PDF files.

    Args:
        files (list): A list of file objects representing PDF files.

    Returns:
        str: The concatenated text content from all the pages of the PDF files.
    """
    raw_text = ""
    if not files: 
        return raw_text
     
    for file in files:
        doc = pymupdf.open(file)
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

def get_vector_store(text_chunks, model="BAAI/bge-m3"):

    if model == "text-embedding-3-large": 
        embeddings = OpenAIEmbeddings(model=model)
    else: 
        embeddings = HuggingFaceEmbeddings(model_name=model)
    
    return FAISS.from_texts(text_chunks, embeddings)

def get_chain(vectorstore, model="gpt-4o"):
    """
    Returns a ConversationalRetrievalChain object initialized with the given parameters.

    Parameters:
    - vectorstore: The vectorstore used as a retriever for the ConversationalRetrievalChain.

    Returns:
    - chain: A ConversationalRetrievalChain object.

    """

    if model.startswith("gpt"):
        llm = ChatOpenAI(model=model)
    else:
        llm = OllamaLLM(model=model) # Make sure to install the model using `ollama install` before using it.
            
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

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