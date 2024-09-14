# `Charta ✨`

`Charta ✨` is a demo of conversational AI that can help you understand the content of your documents. Upload your PDFs and ask questions about the content.


## How to run the project? 
1- Create a virtual python environment:
```python
python3 -m venv env
```
2- Activate the created environment:
```python
source env/bin/activate
```
3- Install the required dependencies:
```python
pip install -r requirements.txt
```
4- Create a `.env` file and add the API keys as follows:
```python
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # placeholder add your real key
HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # placeholder add your real key
```
5- Run the app:
```python
streamlit run app.py
```

Upload your files and start chating with them 🙂
