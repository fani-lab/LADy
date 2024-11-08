"""Runs the implicit dataset generation (DSG) pipeline."""
import argparse
from params import settings
from cmn import review, semeval
from dsg import eval, filter, generate, llm

def load(params):
    """Loads the base dataset of reviews for filtering and labeling"""
    reviews = semeval.SemEvalReview.load(params["path"], latent=params["implicit"])
    review_dicts = [review.to_dict()[0] for review in reviews]
    return review_dicts

def generate_dataset(params):
    """Main driver of the pipeline"""

    # Load the AI client
    client = llm.LLM(params)

    # Load the base dataset
    base_review_set = load(params)

    # Filter the set for only implicit aspect terms
    if params["implicit"]:
        filtered_review_set = filter.filter_implicit(params, base_review_set)
        ground_truth = None
    else:
        filtered_review_set = base_review_set # Implicit aspects are already filtered out
        ground_truth = [review["aos"][0][0][0][0] for review in filtered_review_set]

    # Leverage generative AI to label the dataset
    labeled_review_set = generate.generate_labels(params, client, filtered_review_set)

    # Save results
    return labeled_review_set, ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments for path, category, and model.")

    parser.add_argument("--path", type=str, default="../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml", help="Path to the file or directory")
    parser.add_argument("--category", type=str, default="restaurant", help="Category name")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--implicit", type=bool, default=True, help="Whether to generate implicit or explicit aspects")

    args = parser.parse_args()

    settings["dsg"]["path"] = args.path
    settings["dsg"]["category"] = args.category
    settings["dsg"]["model"] = args.model
    settings["dsg"]["implicit"] = args.implicit

    params = settings["dsg"]
    generate_dataset(params)
