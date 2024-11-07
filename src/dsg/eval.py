"""Evaluation utilities for the pipeline."""
import random

import dotenv
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

def precision(scores):
    """Compute precision (out of all predicted, how many are correct)."""
    return len([score for score in scores if score["similar"]]) / len(scores)

def recall(scores):
    """Compute recall (out of all correct, how many were predicted)."""
    term_set = set([score["source"] for score in scores])
    predicted_count = 0
    for term in term_set:
        if any([score["similar"] for score in scores if score["source"] == term]):
            predicted_count += 1
    return predicted_count / len(term_set)

def eval_aspect_labels(params, source, target):
    """Evaluates predicted labels against ground truth."""

    # Use OpenAI API to compute embeddings
    openai_client = OpenAI(
        api_key=params["openai_api_key"]
    )

    scores = []
    for source, target in zip(source, target):

        # Compute embeddings for prediciton, target pairs
        src_embed = openai_client.embeddings.create(
            input=source,
            model=params["eval"]["embedding_model"],
        ).data[0].embedding

        tgt_embed = openai_client.embeddings.create(
            input=target,
            model=params["eval"]["embedding_model"],
        ).data[0].embedding

        src_embed = np.array(src_embed).reshape(1, -1)
        tgt_embed = np.array(tgt_embed).reshape(1, -1)

        # Compute cosine similarity
        similarity = cosine_similarity(src_embed, tgt_embed)[0][0]
        
        # Log score
        scores.append(
            {
                "source": source,
                "target": target,
                "similarity": similarity,
                "similar": similarity >= params["eval"]["similarity_threshold"],
                "exact_match": source.lower() == target.lower(),
            }
        )

    precision_score = precision(scores)
    recall_score = recall(scores)

    return {
        "raw_scores": scores,
        "precision": precision_score,
        "recall": recall_score,
        "f1_score": 2 * (precision_score * recall_score) / (precision_score + recall_score),
        "exact_match": len([score for score in scores if score["exact_match"]]) / len(scores),
    }

def subset(dataset, n):
    """Return a subset of the dataset for human review."""
    random.shuffle(dataset)
    return dataset[:n]
