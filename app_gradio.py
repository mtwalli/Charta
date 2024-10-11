import gradio as gr
import random
import time


def upload_files(files):
    return [file.name for file in files]

def user(user_message, history: list):
     return "", history + [{"role": "user", "content": user_message}]

def bot(history: list):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    history.append({"role": "assistant", "content": ""})
    for character in bot_message:
        history[-1]['content'] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    gr.Markdown("# Charta ✨")
    gr.Markdown("Charta ✨ is a conversational AI that can help you navigate through your documents. Upload your PDFs and start chatting with them.")

    with gr.Row():
        with gr.Column(scale=1):
            files_output = gr.File(interactive=True, label="Upload your PDFs here", file_count="multiple")
            text_output = gr.Textbox(label="Raw Text", placeholder="Text will appear here...",inputs=files_output)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="✨",type="messages")
            msg = gr.Textbox(label="Ask a question", lines=1, placeholder="Type here...")
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then( bot, chatbot, chatbot)
    
demo.launch()