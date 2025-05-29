import openai
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import logging


class LLMconfig:
    def __init__(self, use_api=True, api_key=None, api_base=None, temperature=0.5, max_tokens=1024, local_model_path=None, model_name=None):
            self.use_api = use_api
            self.api_key = api_key
            self.api_base = api_base
            self.temperature=temperature
            self.max_tokens=max_tokens
            self.local_model_path = local_model_path
            self.model_name = model_name
            
            if self.use_api and not self.api_key: raise ValueError("API key must be provided if use_api is True.")

            if not self.use_api and not self.local_model_path: raise ValueError("Local model path must be provided if use_api is False.")

class LLMHandler:
   
    
    def __init__(self, config):
        self.config = config
        self.generator = None
        
        if self.config.use_api:
            openai.api_key = self.config.api_key
            if self.config.api_base:
                openai.api_base = self.config.api_base
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.local_model_path)
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            
    def _format_api_prompt(self, prompt: str):
        return [
            {"role": "system", "content": "You are an assistant who finds aspects for a given review."},
            {"role": "user", "content": prompt}
        ]
        
    def get_response(self, prompt: str) -> str:
        
        try:
            if self.config.use_api:
                response = openai.ChatCompletion.create(
                    model=self.config.model_name,
                    messages=self._format_api_prompt(prompt),
                    max_tokens=self.config.max_tokens,
                    temperature= self.config.temperature
                )
                return response['choices'][0]['message']['content'].strip()
            else:
                #result = self.generator(prompt, max_length=max_tokens + len(prompt.split()), do_sample=True)
                max_len = min(self.config.max_tokens + len(prompt.split()), 1024)

                result = self.generator(prompt, max_length=max_len, truncation=True, do_sample=True)

                return result[0]['generated_text'].strip()
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return ""
            
        