import gradio as gr
import random
import time
from dotenv import load_dotenv
from helpers import (get_pdf_text,
                     get_text,
                     get_text_chunks, 
                     get_vector_store, 
                     get_chain,
                     get_data_frame)
from  dotenv import load_dotenv


chain = None

def process_pdf(files,progress_gr=gr.Progress()):
    progress_gr(progress=0.25,desc="Reading text")
    text = get_text(files)
   
    progress_gr(progress=0.50,desc="Chuncking text")
    chuncks = get_text_chunks(text)
    
    progress_gr(progress=0.75,desc="Embedding text")
    vector_store = get_vector_store(chuncks)
    
    progress_gr(progress=1,desc="Preparing chain")
    
    global chain
    chain = get_chain(vector_store)

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
            file_uplaod = gr.File(interactive=True, label="Documents", file_count="multiple")
            process = gr.Button("Process",size="sm", variant="primary")
            process.click(process_pdf,inputs=file_uplaod, outputs= gr.Text(label="Processing status"))

        with gr.Column(scale=2):
            gr.ChatInterface(
                fn=bot,
                type="messages",    
                examples=[
                    "What is the document about?",
                    "Give me a short summary?",
                    "Give me the main the topics?"
                    ],
                    chatbot=gr.Chatbot(label="Charta ✨",type="messages"),
                    )

if __name__ == "__main__":
    load_dotenv()
    demo.launch()