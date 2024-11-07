"""Filter stage of the pipeline, extracts reviews with only implicit aspects."""
# For now, this stage does little as we are testing methodologies on the ABSA dataset.
# In the SemEval dataset, reviews are marked with NULL aspects, which are implicit aspects.
# So there is no need for an advanced filter stage at this time.

def filter_implicit(params, dataset):
    """Filter the dataset for only reviews with implicit aspects"""
    filtered_dataset = []
    for review in dataset:
        for triplet in review["aos"]:
            if triplet[0][0][0] == "null":
                filtered_dataset.append(review)
                break
    return filtered_dataset
