
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatLlamaCpp
load_dotenv()

print("Loading model (this will take a few minutes if downloading)...")


llm = ChatLlamaCpp(
    model_path="C:\\Users\\Administrator\\Desktop\\GenerativeAI\\qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    temperature=0.9,
    n_ctx=2048,
    max_tokens=512,
    n_threads=4,
    verbose=True
    
)

print("Chat initialized. Type 'exit' to quit.")

chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    human_message = HumanMessage(content=user_input)
    chat_history.append(human_message)
    response = llm.invoke(chat_history)
    
    print("AI:", response.content)
 
    chat_history.append(response)