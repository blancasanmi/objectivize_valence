
import json

from mistralai import Mistral
from dotenv import load_dotenv
import os

PROMPT = f"""Vous analysez des phrases en français.

Analysez les deux phrases suivantes et retournez un objet JSON avec la structure suivante :

Retournez UNIQUEMENT un objet JSON avec exactement cette structure :
{{
    "sentence_1": {{
        "valence": <nombre décimal entre -1 (très négatif) et 1 (très positif)>,
        "arousal": <nombre décimal entre -1 (faible activation émotionnelle) et 1 (forte activation émotionnelle)>,
        "uncertainty": <nombre décimal entre -1 (très certain) et 1 (très incertain)>
    }},
    "sentence_2": {{
        "valence": <nombre décimal entre -1 (très négatif) et 1 (très positif)>,
        "arousal": <nombre décimal entre -1 (faible activation émotionnelle) et 1 (forte activation émotionnelle)>,
        "uncertainty": <nombre décimal entre -1 (très certain) et 1 (très incertain)>
    }},
    "pair": {{
        "resolution": <nombre décimal entre -1 (incertitude résolue) et 1 (incertitude non résolue)>
    }}
}}"""

load_dotenv()
key = os.getenv("MISTRAL_API_KEY")
print(repr(key))

class MistralModel:
    def __init__(self, model_name: str):
        self.key = os.getenv("MISTRAL_API_KEY")
        self.model_name = model_name

    def query(self, sentence_1: str, sentence_2: str) -> str:       
        with Mistral(
            api_key=self.key,
        ) as mistral:

           res = mistral.chat.complete(
            model=self.model_name,
            random_seed=42,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT + f"\n\nPhrase 1: {sentence_1}\nPhrase 2: {sentence_2}",
                },
            ],
            stream=False,
            response_format={
                "type": "json_object",  
            }
        )

        # Handle response
        return res.choices[0].message.content
    
    def score(self, sentence_1: str, sentence_2: str) -> dict:
        response = self.query(sentence_1, sentence_2)
        return json.loads(response)