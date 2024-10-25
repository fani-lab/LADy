"""Generation stage of the pipeline, generates implicit aspects from reviews."""

def _generate_label(client, model, system_prompt, user_prompt):
    """Given a reviews text, generate an appropriate aspect label."""
    label = client.chat.completions.create(
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
        model=model,
    ).choices[0].message.content.strip()
    return label

def generate_labels(params, dataset, category):
    SENTIMENT_MAP = {
        "+1": "positive",
        "0": "neutral",
        "-1": "negative"
    }
    for i, review in enumerate(dataset):
        label = _generate_label(
            params["openai_client"],
            params["model"],
            params["sys_prompt_label"],
            params["user_prompt_label"] % (review["text"], category, SENTIMENT_MAP[review["aos"][0][0][2]]),
        )
        dataset[i]["aos"][0][0][0][0] = label
    return dataset
    