
import json

from mistralai import Mistral
from dotenv import load_dotenv
import os

PROMPT = f"""You are analyzing French sentences from a psychological experiment on suspense and reading.

Analyze the following two sentences and return a JSON object with the following structure:

Return ONLY a JSON object with this exact structure:
{{
    "sentence_1": {{
        "sentiment": <float between -1 (strongly negative) and 1 (strongly positive)>,
        "valence": <float between 0 (strongly low) and 1 (strongly high)>
    }},
    "sentence_2": {{
        "sentiment": <float between -1 (strongly negative) and 1 (strongly positive)>,
        "valence": <float between 0 (strongly low) and 1 (strongly high)>
    }},
    "pair": {{
        "uncertainty": <0 (no uncertainty), 1 (uncertainty resolved), or 2 (uncertainty unresolved)>
    }}
}}"""

load_dotenv()

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
                    "content": f"sentence_1: {sentence_1}\nsentence_2: {sentence_2}" + PROMPT,
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