from flask import Flask
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

app.config.from_pyfile('settings.py')

@app.route("/")
def index():
    return "Hello world"

@app.post('/translate')
def translate():
    article_en = request.form['original_text']
    translate_code = request.form['translate_code']
    access_token = app.config["HUGGINGFACE_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained("SnypzZz/Llama2-13b-Language-translate", token=access_token)
    model = AutoModelForSeq2SeqLM.from_pretrained("SnypzZz/Llama2-13b-Language-translate", token=access_token)

    model_inputs = tokenizer(article_en, return_tensors="pt")

    # translate from English
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[translate_code]
    )
    translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_sentence[0]