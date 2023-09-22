"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import os
from datetime import datetime

# Import pynecone.
import openai
import pynecone as pc
from chatbot.langchain_model import generate_answer
from pynecone.base import Base


os.environ["OPENAI_API_KEY"] = open("../apikey.txt", "r").read()

parallel_example = {
    "한국어": ["오늘 날씨 어때", "딥러닝 기반의 AI기술이 인기를끌고 있다."],
    "영어": ["How is the weather today", "Deep learning-based AI technology is gaining popularity."],
    "일본어": ["今日の天気はどうですか", "ディープラーニングベースのAIテクノロジーが人気を集めています。"]
}


def translate_text_using_chatgpt(text, src_lang, trg_lang) -> str:
    # fewshot 예제를 만들고
    def build_fewshot(src_lang, trg_lang):
        src_examples = parallel_example[src_lang]
        trg_examples = parallel_example[trg_lang]

        fewshot_messages = []

        for src_text, trg_text in zip(src_examples, trg_examples):
            fewshot_messages.append({"role": "user", "content": src_text})
            fewshot_messages.append({"role": "assistant", "content": trg_text})

        return fewshot_messages

    # system instruction 만들고
    system_instruction = f"assistant는 번역앱으로서 동작한다. {src_lang}를 {trg_lang}로 적절하게 번역하고 번역된 텍스트만 출력한다."

    # messages를만들고
    fewshot_messages = build_fewshot(src_lang=src_lang, trg_lang=trg_lang)

    messages = [{"role": "system", "content": system_instruction},
                *fewshot_messages,
                {"role": "user", "content": text}
                ]

    # API 호출
    response = generate_answer(text)
    translated_text = response['choices'][0]['message']['content']
    # Return
    return translated_text


class Message(Base):
    original_text: str
    text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    def post(self):
        self.messages = self.messages + [
                            Message(
                                original_text=self.text,
                                text=generate_answer(self.text),
                                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                            )
                        ]


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Kakao Developer Helper BOT 🗺", font_size="2rem"),
        pc.text(
            "카카오톡 소셜 API, 카카오톡 채널 API, 카카오싱크에 대한 질문에 답변해드립니다",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            pc.box(
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",

        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        pc.input(
            placeholder="카카오톡 소셜 API, 카카오톡 채널 API, 카카오싱크에 대한 질문에 답변해드립니다",
            on_change=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("질문하기", on_click=State.post, margin_top="1rem"),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Kakao Developer Helper BOT")
app.compile()
