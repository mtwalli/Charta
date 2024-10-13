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
    chain = get_chain(vector_store)

    return "Processed"     

def user(user_message, history: list):
     return "", history + [{"role": "user", "content": user_message}]

def bot(history: list):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    history.append({"role": "assistant", "content": ""})
    for character in bot_message:
        history[-1]['content'] += character
        time.sleep(0.05)
        yield history


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Charta ✨")
    gr.Markdown("Charta ✨ is a conversational AI that can help you navigate through your documents. Upload your PDFs and start chatting with them.")

    with gr.Row():
        with gr.Column(scale=1):
            file_uplaod = gr.File(interactive=True, label="Documents", file_count="multiple")
            process = gr.Button("Process",size="sm", variant="primary")
            process.click(process_pdf,inputs=file_uplaod, outputs= gr.Label(label="Processing status"))

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="✨",type="messages")
            msg = gr.Textbox(label="Ask a question", lines=1, placeholder="Type here...")
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then( bot, chatbot, chatbot)

if __name__ == "__main__":
    load_dotenv()
    demo.launch()