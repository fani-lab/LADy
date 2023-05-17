import argparse
import pandas as pd
import nltk
from rouge import Rouge

def tokenize(raw_corpus, nested):
    corpus = []
    for sentence in raw_corpus:
        # each sentence is one reference/hypothesis
        corpus.append([sentence.split()] if nested else sentence.split())
    return corpus

def load(ref_path, can_path, need_token, r_flag, c_flag):
    ref_raw = pd.read_csv(f'{ref_path}')['sentences'].tolist()
    can_raw = pd.read_csv(f'{can_path}')['sentences'].tolist()
    assert len(ref_raw) == len(can_raw), "Datasets should have the same length"

    ref_corpus = tokenize(ref_raw, r_flag) if need_token else ref_raw
    can_corpus = tokenize(can_raw, c_flag) if need_token else can_raw
    assert len(ref_corpus) == len(can_corpus), "Tokenized corpus should have the same length"

    return ref_corpus, can_corpus

def avg_sentence_tokens(name, path, lan):
    # calculate the average number of sentences
    corpus = pd.read_csv(f'{path}')['sentences'].tolist()
    num_sentences = len(corpus)

    sum_tokens = 0
    for sentence in corpus:
        sum_tokens += len(sentence.split() if not lan == '.zho' else list(sentence))
    
    return [name, num_sentences, sum_tokens, sum_tokens / num_sentences]

def calculate_bleu(ref_path, can_path):
    ref_corpus, can_corpus = load(ref_path, can_path, True, True, False)
    score = nltk.translate.bleu_score.corpus_bleu(ref_corpus, can_corpus) # default 4-gram
    return score

def calculate_rouge(ref_path, can_path):
    rouge = Rouge()
    ref_corpus, can_corpus = load(ref_path, can_path, False, False, False)
    score = rouge.get_scores(can_corpus, ref_corpus, avg=True)['rouge-l']['f'] # uses the ROUGE-L metric
    return score
    
def calculate_em(ref_path, can_path):
    ref_corpus, can_corpus = load(ref_path, can_path, False, False, False)
    sum_em = 0
    for i in range(len(ref_corpus)):
        if can_corpus[i] == ref_corpus[i]:
            sum_em = sum_em + 1
    
    score = sum_em / len(ref_corpus)
    return score

    # produces the same result
    '''
    ref_corpus, can_corpus = load(ref_path, can_path, False, False, False)
    score = accuracy_score(ref_corpus, can_corpus)
    return score
    '''

def metrics(name, ref_data, can_data):
    metrics = [name]
    metrics.append(calculate_bleu(ref_data, can_data))
    metrics.append(calculate_rouge(ref_data, can_data))
    metrics.append(calculate_em(ref_data, can_data))
    return metrics

def main(args):
    tokenMetrics = []

    for d in args.data:
        score_metrics = []
        print(f'=================={d}========================')
        og_data = f'../../output/augmentation-R-{d}/back-translation/D.csv' # Original Dataset
        tokenMetrics.append(avg_sentence_tokens(f'R-{d} D', og_data, "eng"))
        for l in args.lan:
            trans_data = f'../../output/augmentation-R-{d}/back-translation/D.{l}.csv' # Translated Dataset
            back_trans_data = f'../../output/augmentation-R-{d}/back-translation/D_{l}.csv' # Back-translated Dataset

            score_metrics.append(metrics(f'D -> D.{l}', og_data, trans_data))
            score_metrics.append(metrics(f'D.{l} -> D_{l}', trans_data, back_trans_data))
            score_metrics.append(metrics(f'D -> D_{l}', og_data, back_trans_data))

            tokenMetrics.append(avg_sentence_tokens(f'R-{d} D.{l}', trans_data, "." + l))
            tokenMetrics.append(avg_sentence_tokens(f'R-{d} D_{l}', back_trans_data, "_" + l))
        
        df = pd.DataFrame(score_metrics, columns=["Name", "BLEU", "ROUGE-L", "EM"])
        df.to_csv(f'../../output/back-translation-metrics/back-translation-metrics-R-{d}.csv', index=None)
    
    avg_df = pd.DataFrame(tokenMetrics, columns=["Name", "Total Sentences", "Total Tokens", "Avg Tokens"])
    avg_df.to_csv(f'../../output/back-translation-metrics/average-sentences-tokens-R-{args.data}.csv', index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Back-translation Metrics')
    #parser.add_argument('--mt',  nargs='+', type=str.lower, required=True, default=['nllb'], help='a list of translator models')
    parser.add_argument('--lan',  nargs='+', type=str, required=True, default='deu', help='a list of desired languages')
    parser.add_argument('--data', nargs='+', type=str, default='16', required=True, help='year of the data (SemEval), e.g., 16 for SemEval 16')
    #parser.add_argument('--output', dest='output', type=str, default='../../output', help='output path, e.g., ../output')
    args = parser.parse_args()

    main(args)