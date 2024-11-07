"""Script that evaluates DSG accross 3 models: Llama, GPT-4o, and Gemini"""
from params import settings
from main_dsg import generate_dataset
from dsg import eval, plot

def llm_experiment():
    """Test Llama, GPT-4o, and Gemini on identifying aspects"""
    params = settings["dsg"]
    # REAL: params["path"] = "../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml"
    params["path"] = "../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml"
    params["category"] = "restaurant"
    params["implicit"] = False # We need ground truth labels for evaluation

    # Test all models
    models = ["gpt-4o-mini"]
    precision_scores = []
    recall_scores = []
    f1_scores = []
    exact_match_scores = []

    for model in models:
        params["model"] = model
        reviews, targets = generate_dataset(params)
        predictions = [review["aos"][0][0][0][0] for review in reviews]
        stats = eval.eval_aspect_labels(params, predictions, targets)

        precision_scores.append(stats["precision"])
        recall_scores.append(stats["recall"])
        f1_scores.append(stats["f1_score"])
        exact_match_scores.append(stats["exact_match"])

    plot.plot_eval_metrics(models, precision_scores, recall_scores, f1_scores, exact_match_scores)

llm_experiment()
