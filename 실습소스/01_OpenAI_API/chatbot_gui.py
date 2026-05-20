from dotenv import load_dotenv
import os
import threading
import tkinter as tk
from tkinter import scrolledtext
from openai import OpenAI

# .env 파일 불러오기
load_dotenv("C:/env/.env")

# 환경 변수 가져오기
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# 색상 · 폰트
COLORS = {
    "window": "#eef2ff",
    "header": "#4f46e5",
    "header_text": "#ffffff",
    "chat_bg": "#ffffff",
    "chat_border": "#c7d2fe",
    "input_bg": "#ffffff",
    "input_border": "#a5b4fc",
    "btn_send": "#4f46e5",
    "btn_send_hover": "#4338ca",
    "btn_quit": "#64748b",
    "btn_quit_hover": "#475569",
    "btn_text": "#ffffff",
    "user_label": "#4f46e5",
    "user_text": "#1e293b",
    "bot_label": "#059669",
    "bot_text": "#334155",
    "error_text": "#dc2626",
    "hint": "#94a3b8",
}

FONT_FAMILY = "맑은 고딕"
FONT_BODY = (FONT_FAMILY, 11)
FONT_LABEL = (FONT_FAMILY, 11, "bold")
FONT_TITLE = (FONT_FAMILY, 15, "bold")
FONT_INPUT = (FONT_FAMILY, 11)
FONT_BTN = (FONT_FAMILY, 10, "bold")
FONT_HINT = (FONT_FAMILY, 9)


def chat_with_gpt(user_message, history):
    history.append({"role": "user", "content": user_message})
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    reply = completion.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenAI Chatbot")
        self.root.geometry("640x560")
        self.root.minsize(480, 400)
        self.root.configure(bg=COLORS["window"])
        self.history = []

        header = tk.Frame(root, bg=COLORS["header"], padx=16, pady=14)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="OpenAI Chatbot",
            font=FONT_TITLE,
            fg=COLORS["header_text"],
            bg=COLORS["header"],
        ).pack(anchor="w")

        tk.Label(
            header,
            text="gpt-4o-mini · 대화 맥락 유지",
            font=FONT_HINT,
            fg="#c7d2fe",
            bg=COLORS["header"],
        ).pack(anchor="w", pady=(4, 0))

        main = tk.Frame(root, bg=COLORS["window"], padx=16, pady=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        bottom = tk.Frame(main, bg=COLORS["window"])
        bottom.grid(row=1, column=0, sticky="ew")

        tk.Label(
            bottom,
            text="'quit' 입력 또는 종료 버튼으로 나갈 수 있습니다.",
            font=FONT_HINT,
            fg=COLORS["hint"],
            bg=COLORS["window"],
        ).pack(anchor="w", pady=(10, 8))

        input_outer = tk.Frame(bottom, bg=COLORS["input_border"], padx=1, pady=1)
        input_outer.pack(fill=tk.X)

        input_frame = tk.Frame(input_outer, bg=COLORS["input_bg"], padx=8, pady=8)
        input_frame.pack(fill=tk.X)
        input_frame.grid_columnconfigure(0, weight=1)

        self.entry = tk.Entry(
            input_frame,
            font=FONT_INPUT,
            bg=COLORS["input_bg"],
            fg=COLORS["user_text"],
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            highlightcolor=COLORS["btn_send"],
            insertbackground=COLORS["user_label"],
        )
        self.entry.grid(row=0, column=0, sticky="ew", ipady=6, padx=(0, 10))
        self.entry.bind("<Return>", lambda _event: self.send_message())
        self.entry.focus()

        self.send_btn = tk.Button(
            input_frame,
            text="전송",
            font=FONT_BTN,
            fg=COLORS["btn_text"],
            bg=COLORS["btn_send"],
            activeforeground=COLORS["btn_text"],
            activebackground=COLORS["btn_send_hover"],
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.send_message,
        )
        self.send_btn.grid(row=0, column=1, padx=(0, 8))

        tk.Button(
            input_frame,
            text="종료",
            font=FONT_BTN,
            fg=COLORS["btn_text"],
            bg=COLORS["btn_quit"],
            activeforeground=COLORS["btn_text"],
            activebackground=COLORS["btn_quit_hover"],
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.root.quit,
        ).grid(row=0, column=2)

        chat_frame = tk.Frame(
            main,
            bg=COLORS["chat_border"],
            highlightbackground=COLORS["chat_border"],
            highlightthickness=1,
        )
        chat_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 0))

        self.chat_area = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            height=12,
            state="disabled",
            font=FONT_BODY,
            bg=COLORS["chat_bg"],
            fg=COLORS["bot_text"],
            relief=tk.FLAT,
            borderwidth=0,
            padx=14,
            pady=12,
            spacing1=2,
            spacing3=6,
            cursor="arrow",
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.chat_area.tag_config("user_label", foreground=COLORS["user_label"], font=FONT_LABEL)
        self.chat_area.tag_config("user_text", foreground=COLORS["user_text"], font=FONT_BODY)
        self.chat_area.tag_config("bot_label", foreground=COLORS["bot_label"], font=FONT_LABEL)
        self.chat_area.tag_config("bot_text", foreground=COLORS["bot_text"], font=FONT_BODY)
        self.chat_area.tag_config("error_text", foreground=COLORS["error_text"], font=FONT_BODY)

        self.append_welcome()

    def append_welcome(self):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, "Bot: ", "bot_label")
        self.chat_area.insert(
            tk.END,
            "안녕하세요! 무엇이든 물어보세요.\n\n",
            "bot_text",
        )
        self.chat_area.config(state="disabled")

    def append_chat(self, role, text):
        self.chat_area.config(state="normal")
        if role == "user":
            self.chat_area.insert(tk.END, "You: ", "user_label")
            self.chat_area.insert(tk.END, f"{text}\n\n", "user_text")
        elif role == "bot":
            self.chat_area.insert(tk.END, "Bot: ", "bot_label")
            self.chat_area.insert(tk.END, f"{text}\n\n", "bot_text")
        else:
            self.chat_area.insert(tk.END, "Bot: ", "bot_label")
            self.chat_area.insert(tk.END, f"{text}\n\n", "error_text")
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def set_input_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.entry.config(state=state)
        self.send_btn.config(state=state)

    def send_message(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return

        self.entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            self.root.quit()
            return

        self.append_chat("user", user_input)
        self.set_input_enabled(False)

        def worker():
            try:
                reply = chat_with_gpt(user_input, self.history)
                self.root.after(0, lambda: self.on_reply(reply))
            except Exception as exc:
                self.root.after(0, lambda: self.on_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def on_reply(self, reply):
        self.append_chat("bot", reply)
        self.set_input_enabled(True)
        self.entry.focus()

    def on_error(self, message):
        self.append_chat("error", f"오류: {message}")
        self.set_input_enabled(True)
        self.entry.focus()


if __name__ == "__main__":
    root = tk.Tk()
    ChatbotApp(root)
    root.mainloop()
