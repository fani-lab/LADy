"""Class for interfacing with a given LLM"""
from openai import OpenAI

class LLM:
    """General class to interface with GPT, Llama, or Gemini"""
    def __init__(self, params):
        self.model_name = params["model"]
        self.client = None

        if self.model_name == "gpt-4o-mini":
            self.client = OpenAI(
                api_key=params["openai_api_key"]
            )

    def generate_completion(self, sys_prompt, usr_prompt):
        """Generates a completion given a prompt"""
        if self.model_name == "gpt-4o-mini":
            return self._generate_completion_openai(sys_prompt, usr_prompt)
        elif self.model_name == "llama":
            return self._generate_completion_llama(sys_prompt)
        elif self.model_name == "gemini":
            return self._generate_completion_gemini(sys_prompt)
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
        raise NotImplementedError
