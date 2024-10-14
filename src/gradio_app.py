import gradio as gr
import time
from dotenv import load_dotenv
from helpers import (get_text,
                     get_text_chunks, 
                     get_vector_store, 
                     get_chain)
from  dotenv import load_dotenv


chain = None

def process_pdf(files,embedding_model="BAAI/bge-m3",llm_mdel="gpt-4o",progress_gr=gr.Progress()):
    print(f"Files:{files}, Embedding model:{embedding_model}, LLM model:{llm_mdel}")
    progress_gr(progress=0.25,desc="Reading text")
    text = get_text(files)
   
    progress_gr(progress=0.50,desc="Chuncking text")
    chuncks = get_text_chunks(text)
    
    progress_gr(progress=0.75,desc="Embedding text")
    vector_store = get_vector_store(chuncks,model=embedding_model)
    
    progress_gr(progress=1,desc="Preparing chain")
    
    global chain
    chain = get_chain(vector_store,model=llm_mdel)

    return "Completed 🎉"     

def bot(message , history):
    response = chain.invoke(message)
    reponse_message = ""
    for character in response['chat_history'][-1].content:
            reponse_message += character
            time.sleep(0.01)
            yield reponse_message 


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Charta ✨")
    gr.Markdown("_Charta_ is a conversational AI that can help you navigate through your documents. Upload your PDFs and start chatting with them.")

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model_choice = gr.Dropdown(["BAAI/bge-m3","text-embedding-3"], label="Embedding model")
            llm_model_choice = gr.Dropdown(["gpt-4o","llama3.1","gemma2:9b"], label="LLM model")
            file_uplaod = gr.File(interactive=True, label="Documents", file_count="multiple")
            status = gr.Text(label="Processing status")
            with gr.Group():
                with gr.Row():
                    clear_btn = gr.Button("Clear",size="sm", variant="secondary")
                    process_btn = gr.Button("Process",size="sm", variant="primary")
    
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Charta ✨",type="messages")
            gr.ChatInterface(
                fn=bot,
                type="messages",    
                examples=[
                    "What is the document about?",
                    "Give me a short summary?",
                    "Give me the main the topics?"
                    ],
                    chatbot=chat,
                    )
        
        process_btn.click(process_pdf,inputs=[file_uplaod,embedding_model_choice,llm_model_choice], outputs=status) 
        clear_btn.click(
            lambda: [
                None,
                "BAAI/bge-m3",
                "gpt-4o",
                None,
                None
                ],
            outputs=[
                file_uplaod,
                embedding_model_choice,
                llm_model_choice,
                status,chat
                ]
            )        

if __name__ == "__main__":
    load_dotenv()
    demo.launch()