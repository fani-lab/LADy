"""Class for interfacing with a given LLM"""
import requests
from openai import OpenAI

class LLM:
    """General class to interface with GPT, Llama, or Gemini"""
    def __init__(self, params):
        self.model_name = params["model"]

        if self.model_name == "gpt-4o-mini":
            self.client = OpenAI(
                api_key=params["openai_api_key"]
            )
        elif self.model_name == "llama":
            pass
        elif self.model_name == "gemini-1.5-flash":
            self.URL = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={params['gemini_api_key']}"
        else:
            print(f"Error: Model {self.model_name} not supported.")
            raise NotImplementedError

    def generate_completion(self, sys_prompt, usr_prompt):
        """Generates a completion given a prompt"""
        if self.model_name == "gpt-4o-mini":
            return self._generate_completion_openai(sys_prompt, usr_prompt)
        elif self.model_name == "llama":
            return self._generate_completion_llama(sys_prompt)
        elif self.model_name == "gemini-1.5-flash":
            return self._generate_completion_gemini(sys_prompt + " " + usr_prompt)
        else:
            return None
    
    def _generate_completion_openai(self, system_prompt, user_prompt):
        """Generates a completion using the OpenAI API"""
        label = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            model=self.model_name,
        ).choices[0].message.content.strip()
        return label
    
    def _generate_completion_llama(self, prompt):
        """Generates a completion using Llama"""
        raise NotImplementedError
    
    def _generate_completion_gemini(self, prompt):
        """Generates a completion using Gemini"""
        
        # Main gemini SDK requires Python 3.9+ and LADy is Python 3.8, so requests are made using HTTP
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(self.URL, headers=headers, json=payload)
            response.raise_for_status()  # Raises an error for bad responses
            result = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return result
        except requests.exceptions.RequestException as e:
            print("Error, unable to generate Gemini response:", e)
            return ""
