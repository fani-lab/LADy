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
metrics = ['P', 'recall', 'ndcg_cut', 'map_cut', 'success']

topkstr = '1,2,5,10,100'
# topkstr = '1,2,3,4,5,6,7,8,9,10,100'
# topkstr = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100'
# topk = '1:1:100'
