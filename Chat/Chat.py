import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn as uvi

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

tokenizer = None
model = None

def load_model():
	global tokenizer
	global model
	tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
	model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.bfloat16)

	return tokenizer, model

def chatbot(tokenizer, model, prompt):
	inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
	input_length = inputs.input_ids.shape[1]
	outputs = model.generate(
		**inputs, max_new_tokens=128, do_sample=False, temperature=0.2, top_p=0.8, top_k=50, return_dict_in_generate=True
	)
	token = outputs.sequences[0, input_length:]
	output_str = tokenizer.decode(token)

	return output_str

app = FastAPI()

@app.get("/")
def read_root():
		return {"Hello": "World"}

@app.get("/chat/{prompt}")
def chat(prompt: str):
		prompt = f"<humnan>: {prompt}/n<bot>:"
		output_str = chatbot(tokenizer, model, prompt)
		if "<humnan>" in output_str:
			output_str = output_str.split("<humnan>")[0]
		return {"response": output_str}

if __name__ == '__main__':
	load_model()
	uvi.run(app, host='0.0.0.0', port=8000)
