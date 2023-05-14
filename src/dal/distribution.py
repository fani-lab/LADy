from collections import Counter
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from nltk.collocations import *

from cmn.semeval import SemEvalReview
   
def get_category(datapath, output, plot=False, plot_title=None):
    try:
        print("Loading the category pickle ...")
        with open(f'{output}/asp_distn.pkl', 'rb') as f:
            stats = pickle.load(f)
            if plot: plot_category(stats, output, plot_title)
        
    except FileNotFoundError:
        stats = {}
        print("File not found! Generating category ...")
        reviews = pd.read_pickle(datapath)
        numReview = len(reviews)
        ncategory_nreviews = Counter()
        for review in reviews:
            ncategory_nreviews.update([review.get_category()])
    
        stats['ncategory_nreviews'] = {k: v / numReview for k, v in 
                                    sorted(ncategory_nreviews.items(), key=lambda item: item[1], reverse=True)}

        if plot: plot_category(stats, output, plot_title)
        return stats

def plot_category(stats, output, plot_title=None):
    print("plotting category data ...")
    for k, v in stats.items():
        fig, ax = plt.subplots()
        ax.bar(v.keys(), v.values())

        ax.set_ylabel('% Review in Dataset')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.get_label().set_size(12)
        ax.yaxis.get_label().set_size(12)

        ax.set_title(plot_title)

        fig.savefig(f'{output}/{k}.pdf', dpi=100, bbox_inches='tight')
        plt.show()

def get_distribution(datapath, output, plot=False, plot_title=None):
    try:
        print("Loading distribution pickles...")
        with open(f'{output}/stats.pkl', 'rb') as infile:
            stats = pickle.load(infile)
            if plot: plot_distribution(stats, output, plot_title)
    except FileNotFoundError:
        print("File not found! Generating category ...")
        stats = {}
        reviews = pd.read_pickle(datapath)

        asp_nreviews = Counter() # aspects : number of reviews that contains the aspect
        token_nreviews = Counter() # tokens : number of reviews that contains the token
        nreviews_naspects = Counter() # x number of reviews with 1 aspect, 2 aspects, ...
        nreviews_ntokens = Counter() # x number of reviews with 1 token, 2 tokens, ...

        for review in reviews:
            r_aspects = review.get_aos()[0]
            r_tokens = [token for sentence in review.get_sentences() for token in sentence]

            asp_nreviews.update(" ".join(a) for (a, o, s) in r_aspects)
            token_nreviews.update(token for token in r_tokens)

            nreviews_naspects.update([len(r_aspects)])
            nreviews_ntokens.update([len(r_tokens)])
        
        naspects_nreviews = Counter(asp_nreviews.values()) # x number of aspects with 1 review, 2 reviews, ...
        ntokens_nreviews = Counter(token_nreviews.values()) # x number of tokens with 1 review, 2 reviews, ...

        stats['nreviews_naspects'] = {k: v for k, v in sorted(nreviews_naspects.items(), key=lambda item: item[1], reverse=True)}
        stats['nreviews_ntokens'] = {k: v for k, v in sorted(nreviews_ntokens.items(), key=lambda item: item[1], reverse=True)}
        stats['naspects_nreviews'] = {k: v for k, v in sorted(naspects_nreviews.items(), key=lambda item: item[1], reverse=True)}
        stats['ntokens_nreviews'] = {k: v for k, v in sorted(ntokens_nreviews.items(), key=lambda item: item[1], reverse=True)}

        # print(nreviews_naspects) # k : number of reviews with k aspects
        # print(nreviews_ntokens) # k : number of reviews with k tokens
        # print(naspects_nreviews) # k : number of aspects appearing in k reviews
        # print(ntokens_nreviews) # k : number of tokens appearing in k reviews
        # print("\n============\n")
        print(stats)
        with open(f'{output}/stats.pkl', 'wb') as outfile: pickle.dump(stats, outfile)
    
        if plot: plot_distribution(stats, output, plot_title)
        return stats

def plot_distribution(stats, output, plot_title):
    print("plotting distribution data ...")
    for k, v in stats.items():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(*zip(*stats[k].items()), marker='x', linestyle='None', markeredgecolor='m')
        ax.set_xlabel(k.split('_')[1][0].replace('n', '#') + k.split('_')[1][1:])
        ax.set_ylabel(k.split('_')[0][0].replace('n', '#') + k.split('_')[0][1:])
        ax.grid(True, color="#93a1a1", alpha=0.3)
        ax.minorticks_off()
        ax.xaxis.set_tick_params(size=2, direction='in')
        ax.yaxis.set_tick_params(size=2, direction='in')
        ax.xaxis.get_label().set_size(12)
        ax.yaxis.get_label().set_size(12)
        ax.set_title(plot_title)
        fig.savefig(f'{output}/{k}.pdf', dpi=100, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    datasets = ['../output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml',
                '../output/semeval+/SemEval-14/Semeval-14-Restaurants_Train.xml',
                '../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml',
                '../output/semeval+/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml'
            ]
    for data in datasets:
        get_distribution(f'{data}/reviews.pkl', data, True, None)
        get_category(f'{data}/reviews.pkl', data, True, None)