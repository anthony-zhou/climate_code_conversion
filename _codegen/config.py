import os
import dotenv
import promptlayer

dotenv.load_dotenv()

# Only use promptlayer if the API key is provided
if os.environ.get("PROMPTLAYER_API_KEY"):
    promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
    openai = promptlayer.openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
else:
    import openai

    openai.api_key = os.environ.get("OPENAI_API_KEY")


model_name = "gpt-4"
