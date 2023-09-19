import pynecone as pc


class ChatBotConfig(pc.Config):
    pass


config = ChatBotConfig(
    app_name="chatbot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)