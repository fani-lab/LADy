from llm.llm_config_handler import LLMconfig, LLMHandler
from cmn.review import Review  
from llm.prompt_builder import PromptBuilder

from tqdm import tqdm
import json
import re
from omegaconf import DictConfig
import pandas as pd
import os
import string



class LLMReviewProcessor:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.llm_handler = self._init_llm_handler()
        self.prompt_builder = PromptBuilder()

    def _init_llm_handler(self):
        config = LLMconfig(
            use_api=self.cfg.llmargs.use_api,
            api_key=self.cfg.llmargs.api_key,
            api_base=self.cfg.llmargs.api_base,
            model_name=self.cfg.llmargs.model_name,
            temperature=self.cfg.llmargs.temperature,
            max_tokens=self.cfg.llmargs.max_tokens
        )
        return LLMHandler(config)

    
    @staticmethod
    def find_aspect_indices(aspect: str, sentence_tokens) :
        aspect_tokens = aspect.lower().split()
        tokens = [token.lower().strip(string.punctuation) for token in sentence_tokens]

        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i + len(aspect_tokens)] == aspect_tokens:
                return list(range(i, i + len(aspect_tokens)))

        return -1  
    
    def process_reviews(self, reviews: list):
        enhanced_reviews = []
        for i, r in enumerate(tqdm(reviews)):
            sample_review = vars(r)
            print(f"_____Review {i}____")
            print(sample_review)

            prompt = self.prompt_builder.build_prompt(sample_review)
            response = self.llm_handler.get_response(prompt)
            matches = re.findall(r'\{.*?\}', response, re.DOTALL)

            if not matches:
                print("No JSON object found in response")
                continue

            all_aspects = []
            seen_aspects = set()
            tokens = [word.strip().lower() for sentences in sample_review["sentences"] for word in sentences]
            updated_aos = []
            sentiment = sample_review.get("aos", [[[-1, [], -1]]])[0][0][2]

            for json_str in matches:
                try:
                    aspect_data = json.loads(json_str)
                    aspect_term = aspect_data.get("aspect", "").strip().lower()
                    if not aspect_term or aspect_term in seen_aspects:
                        continue
                    seen_aspects.add(aspect_term)
                    aspect_indices = self.find_aspect_indices(aspect_term, tokens)
                    updated_aos.append((aspect_indices if aspect_indices != -1 else [-1], [], sentiment))
                    all_aspects.append(aspect_term)
                except json.JSONDecodeError:
                    print("Failed to parse JSON:", json_str)

            updated_review = Review(
                id=sample_review["id"],
                sentences=sample_review["sentences"],
                time=sample_review["time"],
                author=sample_review["author"],
                lempos=sample_review.get("lempos", ""),
                aos=[updated_aos],
                parent=sample_review.get("parent", None),
                lang=sample_review.get("lang", "eng_Latn"),
                implicit=sample_review.get("implicit", False)
            )
            updated_review.aspects = all_aspects
            enhanced_reviews.append(updated_review)

        self.save_to_pickle(enhanced_reviews)
        return enhanced_reviews
    
    def save_to_pickle(self, reviews):
        output_path = self.cfg.llmargs.output
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.basename(self.cfg.args.data)
        print(f'\nSaving processed Review.pickle file {output_dir}/{filename}...')
        pd.to_pickle(reviews, os.path.join(output_dir, filename))


# Entry function
def mainllm(cfg: DictConfig, reviews: list):
    processor = LLMReviewProcessor(cfg)
    return processor.process_reviews(reviews)

if __name__ == "__main__":
    mainllm()
    

    


