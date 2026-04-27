import os
#import google.generativeai as genai
from typing import List
from groq import Groq
import traceback
from dotenv import load_dotenv

load_dotenv()
class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Clé API manquante dans .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Erreur LLM: {e}")
            return "Désolé, une erreur technique est survenue."
        
class LLamaService():
    
    def __init__(self,  model_name="openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("API Key GROQ manquante.")
        
        try:
            self.model = model_name
            self.client = Groq(api_key=api_key)
            print("Client Groq initialisé avec succès.")
        except Exception as e:
            print(f"Erreur lors de l'initialisation du client Groq: {e}")
        
    def generate(self, prompt: str, stop: List[str] = None) -> str:
        if not self.client:
            return "Le client Groq n'est pas disponible."

        if not prompt:
            return "Prompt manquant"

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                # On utilise un modèle garanti d'être actif et performant
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
            )

            evaluation = chat_completion.choices[0].message.content
            
            return evaluation.strip()

        except Exception as e:
            print("-- ERREUR DÉTAILLÉE LORS DE L'APPEL API GROQ --")
            traceback.print_exc()
            return f"Erreur lors de la communication avec l'API Groq: {str(e)}"