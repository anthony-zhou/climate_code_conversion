import os
import dotenv
import promptlayer

dotenv.load_dotenv()

promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
openai = promptlayer.openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

model_name = "gpt-3.5-turbo"
