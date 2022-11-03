seed = 0

# aspect modeling settings
no_extremes = {
    'no_below': 10,    # happen less than no_below number in total
    'no_above': 0.9,    # happen in no_above percent of reviews
}
doctype = 'snt' # 'rvw' ==> if 'rvw': review = [[review]] else if 'snt': review = [[subreview1], [subreview2], ...]
iter_c = 100
cores = 0
nwords = 20
qualities = ['Coherence', 'Perplexity']
# training settings
train_ratio = 0.85 # 1 - train_ratio goes to test
nfolds = 5 # on the train, nfold x-valid

# evaluation settings
metrics = ['success', 'P', 'recall', 'ndcg_cut', 'map_cut']

topkstr = '1,2,5,10,100'

topk = 10
