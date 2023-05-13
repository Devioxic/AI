from fastapi import FastAPI
import uvicorn as uvi
from vertexai.preview.language_models import ChatModel

app = FastAPI()

def chatbot(prompt, temperature=0.2):

    chat_model = ChatModel.from_pretrained("chat-bison@001")

    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 512,    # Token limit determines the maximum amount of text output.
        "top_p": 0.9,               # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,                 # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    chat = chat_model.start_chat(
        context="You are an assistant named Ruby",
        examples=[
        ]
    )

    response = chat.send_message(prompt, **parameters)
    print(response.text)

    return response



@app.get("/")
def read_root():
		return {"Hello": "World"}

@app.get("/chat/{prompt}")
def chat(prompt: str):
		output_str = chatbot(prompt)
		return {"response": output_str}

if __name__ == '__main__':
	uvi.run(app, host='0.0.0.0', port=8000)
