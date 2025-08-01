from llm.llm_config_handler import LLMconfig, LLMHandler
from cmn.review import Review  
from llm.prompt_builder import PromptBuilder
from llm.trustworthiness_check import TrustWorthiness_PromptBuilder, model_Evaluator
from tqdm import tqdm
import json
import re
from omegaconf import DictConfig
import pandas as pd
import os
import string
import random



class LLMReviewProcessor:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.llm_handler = self._init_llm_handler()
        self.prompt_builder = PromptBuilder(top_k=cfg.llmargs.top_k_aspects)

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
            if tokens[i:i + len(aspect_tokens)] == aspect_tokens: return list(range(i, i + len(aspect_tokens)))
        return -1  
    
    def process_reviews(self, reviews: list):
        enhanced_reviews = []
        for i, r in enumerate(tqdm(reviews)):
            sample_review = vars(r)        
            if sample_review.get('implicit', [False])[0] is not True: continue

            prompt = self.prompt_builder.build_prompt(sample_review)
            
            max_retries = 5
            valid_json_found = False
            matches = []
            
            for attempt in range(max_retries):
                response = self.llm_handler.get_response(prompt)
                matches = re.findall(r'\{.*?\}', response, re.DOTALL)

                for json_str in matches:
                    try:
                        aspect_data = json.loads(json_str)
                        if "aspect" in aspect_data and aspect_data["aspect"]:
                            valid_json_found = True
                            break
                    except json.JSONDecodeError: continue

                if valid_json_found: break
                else: print(f"Invalid or no valid JSON with 'aspect' found. Attempt {attempt + 1} of {max_retries}")

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
                    aspect = aspect_data.get("aspect", "")
                    if isinstance(aspect, list) and aspect: aspect = aspect[0]  # or join if multiple items are expected
                    aspect_term = aspect.strip().lower()
                    
                    if not aspect_term or aspect_term in seen_aspects: continue

                    seen_aspects.add(aspect_term)
                    aspect_indices = self.find_aspect_indices(aspect_term, tokens)
                    updated_aos.append((aspect_indices if aspect_indices != -1 else [-1], [], sentiment))
                    all_aspects.append(aspect_term)
                except json.JSONDecodeError: print("Failed to parse JSON:", json_str)

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

        if(len(enhanced_reviews) > 0):
            self.save_to_pickle(enhanced_reviews)
            return enhanced_reviews
        else:
            print("No Implicit Review to be Processed ")
            return None

    
    def save_to_pickle(self, reviews):
        output_path = self.cfg.llmargs.output
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        filename = os.path.basename(self.cfg.args.data)
        print(f'\nSaving processed Review.pickle file {output_dir}/{filename}...')
        pd.to_pickle(reviews, os.path.join(output_dir, filename))

# Model Evalution Code

    
    def evaluate_llm_trustworthiness(self, reviews: list):
        prompt_builder = TrustWorthiness_PromptBuilder()
        results = []

        for i, review in enumerate(tqdm(reviews)):
            print(vars(review))
            if not hasattr(review, 'aspects') or not review.aspects:
                review.aspects = self.extract_aspects_from_aos(review)
                print(review.aspects)
                print()
                if not review.aspects: continue

            review_text = ' '.join(review.sentences[0])
            review_dict = vars(review)

            # Correct aspect
            correct_aspect = random.choice(review.aspects)
            correct_prompt = prompt_builder.build_prompt(review_dict, correct_aspect)
            correct_response = self.llm_handler.get_response(correct_prompt)
            correct_pred = self._parse_llm_answer(correct_response)

            results.append({
                "review_id": review.id,
                "review_text": review_text,
                "aspect": correct_aspect,
                "expected_answer": "Yes",
                "model_prediction": correct_pred,
                "is_correct": correct_pred == "Yes"
            })

            #  Incorrect aspect
            wrong_aspect = self._get_wrong_aspect(reviews, i)
            if not wrong_aspect:
                continue

            wrong_prompt = prompt_builder.build_prompt(review_dict, wrong_aspect)
            wrong_response = self.llm_handler.get_response(wrong_prompt)
            wrong_pred = self._parse_llm_answer(wrong_response)

            results.append({
                "review_id": review.id,
                "review_text": review_text,
                "aspect": wrong_aspect,
                "expected_answer": "No",
                "model_prediction": wrong_pred,
                "is_correct": wrong_pred == "No"
            })

        self.save_to_excel(results)
        return True
       
        
    def accuracy_Evaluator(self):
        path = self.cfg.llmargs.outputEval
        model_eval = model_Evaluator()
        
        if os.path.exists(path) and os.path.isdir(path):
            model_eval.evaluator(path)
        else:
            print("The specified path does not exist or is not a directory.")
    
    def save_to_excel(self, results):
        output_path = self.cfg.llmargs.outputEval
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        base_name = os.path.splitext(os.path.basename(self.cfg.args.data))[0]
        filename = f"{base_name}_trustworthiness_eval.xlsx"

        df = pd.DataFrame(results)
        df.to_excel(os.path.join(output_dir, filename), index=False)
        print(f'\Saved processed Evaluation  file {output_dir}/{filename}...')
        
   
    def _parse_llm_answer(self, response: str) -> str:
        match = re.findall(r'Answer\s*:\s*({\s*"Answer"\s*:\s*"(Yes|No)"\s*})', response, re.IGNORECASE)
        if match:
            last_json_str = match[-1][0]  # Full JSON string
            try:
                answer_dict = json.loads(last_json_str)
                return answer_dict.get("Answer", "Invalid").capitalize()
            except json.JSONDecodeError: return "Invalid"
        return "Invalid"
    
    def _get_wrong_aspect(self, reviews, exclude_index):
        other_reviews = [r for idx, r in enumerate(reviews) if idx != exclude_index and hasattr(r, 'aspects') and r.aspects]
        if not other_reviews: return None
        return random.choice(random.choice(other_reviews).aspects)

    def extract_aspects_from_aos(self, review):
        aspects = []
        for ao_list in review.aos:
            for aspect_info in ao_list:
                indices = aspect_info[0]  # aspect token indices
                if indices and indices[0] != -1:
                    flat_tokens = [word for sent in review.sentences for word in sent]
                    try:
                        aspect_term = ' '.join(flat_tokens[i] for i in indices if i < len(flat_tokens))
                        if aspect_term.strip():
                            aspects.append(aspect_term.strip().lower())
                    except IndexError: continue
        return aspects

# Entry function
def mainllm(cfg: DictConfig, reviews: list):
    processor = LLMReviewProcessor(cfg)
    return processor.process_reviews(reviews)

def llmEval(cfg: DictConfig, reviews: list):
    processor = LLMReviewProcessor(cfg)
    
    bolVal = processor.evaluate_llm_trustworthiness(reviews)
    if(bolVal): processor.accuracy_Evaluator()
    


    


    

    


