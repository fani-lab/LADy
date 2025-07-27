
class PromptBuilder:
    def __init__(self, task_description=None, top_k=1):
        self.task_description = task_description or (
            "Identify the latent aspect targeted by the sentiment in the review. "
            "If the aspect is explicitly mentioned, return its index; if it's implicit, return the inferred aspect and use index -1."
        )
        self.top_k = top_k

    def build_prompt(self, review_entry: dict) -> str:
        review_text = ' '.join(review_entry['sentences'][0])
        prompt = (
            f"Review: \"{review_text}\"\n"
            f"Task: {self.task_description}\n"
            f"Return exactly the top {self.top_k} aspect(s) that best represent the sentiment in this review.\n"
            f"Output Format: {{\"aspect\": \"<aspect_name>\", \"index\": <index_list or -1>}}"
        )
        
        return prompt
