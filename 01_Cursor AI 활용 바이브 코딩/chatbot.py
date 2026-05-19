from dotenv import load_dotenv
import os
from openai import OpenAI

# .env 파일 불러오기
load_dotenv("C:/env/.env")

# 환경 변수 가져오기
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)


def chat_with_gpt(user_message, history):
    history.append({"role": "user", "content": user_message})
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    reply = completion.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


history = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "quit":
        break
    print("Bot:", chat_with_gpt(user_input, history))
