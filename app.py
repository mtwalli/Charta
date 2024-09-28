import streamlit as st
from dotenv import load_dotenv
from helpers import (get_pdf_text,
                     get_text_chunks, 
                     get_vector_store, 
                     get_chain,
                     get_data_frame)

def answer_question(user_question):
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
            avatar = None
            if 'HumanMessage' in str(type(message)): 
                role = "human" 
                avatar = "user"
            else: 
                role = "assistant"
                avatar = "ai"

            with st.chat_message(name=role, avatar=f"assets/{avatar}.png"):
                st.markdown(message.content)

def main():
    load_dotenv()
    st.set_page_config(page_title="Charta", page_icon=":sparkles:",layout="wide")
    
    if "chain" not in st.session_state: 
        st.session_state.chain = None

    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = None
    
    st.header("Charta :green[`>`] ")
    st.markdown("_Charta :green[`>`] is a conversational AI that can help you navigate through your documents. Upload your :violet[PDFs] and start chatting with :violet[them]._")
    
    with st.sidebar:
        st.subheader("Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here ", accept_multiple_files=True, type=["pdf"])
        
        if len(pdf_docs) != 0 and st.session_state.chain is None:
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs).strip()
                st.text_area("Raw Text", raw_text)
                
                # get the text chunks
                text_chunks = [text.strip() for text in get_text_chunks(raw_text)]
                
                st.write("Text Chunks:")
                st.write(text_chunks)

                # create vector store
                if len(text_chunks) == 0:
                    st.error("No text chunks found. Please upload a valid PDF file.")
                    return

                vectorstore = get_vector_store(text_chunks)
                st.write("Vector Store:")
                st.write(get_data_frame(vectorstore))

                # create chain
                st.session_state.chain = get_chain(vectorstore)

  
    if st.session_state.chain:
        st.subheader("Chat with Charta")
        user_question = st.chat_input("Ask a question")
        if user_question:
            with st.spinner("Thinking..."):
                answer_question(user_question)          

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")