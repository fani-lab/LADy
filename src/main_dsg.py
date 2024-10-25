"""Runs the implicit dataset generation (DSG) pipeline."""
import dotenv
from openai import OpenAI
from params import settings
from cmn import review, semeval
from dsg import eval, filter, generate

def load(path="../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml"):
    """Loads the base dataset of reviews for filtering and labeling"""
    reviews = semeval.SemEvalReview.load(path, latent=True)
    review_dicts = [review.to_dict()[0] for review in reviews]
    return review_dicts

def generate_dataset(params, category):
    """Main driver of the pipeline"""
    # Load the OpenAI API key
    dotenv.load_dotenv()
    
    # Load the OpenAI client
    openai_client = OpenAI(
        api_key=params["openai_api_key"]
    )
    params["openai_client"] = openai_client

    # Load the base dataset
    base_review_set = load()

    # Filter the set for only implicit aspect terms
    filtered_review_set = filter.filter_dataset(params, base_review_set)

    # Leverage generative AI to label the dataset
    labeled_review_set = generate.generate_labels(params, filtered_review_set, category)

    # Save results
    for labeled_review in labeled_review_set:
        print(f"{labeled_review}\n")

if __name__ == "__main__":
    params = settings["dsg"]
    category = "restaurant" # For now, test on restaurant reviews, but this will be a CLI arg in the future
    generate_dataset(params, category)
