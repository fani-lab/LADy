"""Generation stage of the pipeline, generates implicit aspects from reviews."""

def _generate_label(llm, system_prompt, user_prompt):
    """Given a reviews text, generate an appropriate aspect label."""
    label = llm.generate_completion(system_prompt, user_prompt)
    return label

def generate_labels(params, llm, dataset):
    SENTIMENT_MAP = {
        "+1": "positive",
        "0": "neutral",
        "-1": "negative"
    }
    for i, review in enumerate(dataset):
        label = _generate_label(
            llm,
            params["sys_prompt_label"],
            params["user_prompt_label"] % (review["text"], params["category"], SENTIMENT_MAP[review["aos"][0][0][2]]),
        )
        dataset[i]["aos"][0][0][0][0] = label
    return dataset
    