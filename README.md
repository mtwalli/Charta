# `Charta ✨`

`Charta ✨` is a demo of conversational AI that can help you understand the content of your documents. Upload your PDFs and ask questions about the content.


## How to run the project? 
1- Create a virtual python environment:
```sh
python3 -m venv env
```
2- Activate the created environment:
```sh
source env/bin/activate
```
3- Install the required dependencies:
```python
pip install -r requirements.txt
```
4- Create a `.env` file and add the API keys as follows:
```.env
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # placeholder add your real key
HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # placeholder add your real key
```
5- Run `streamlit` app:
```sh
streamlit run src/streamlit_app.py
```

6- Run `gradio` app:
```sh
gradio src/gradio_app.py
```

Upload your files and start chating with them 🙂

> [!NOTE]  
> To use open-source models via [Ollama](https://ollama.com/), first install Ollama, and then download the necessary models.
