import argparse, os
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from simalign import SentenceAligner


def load(path1, path2):
    df1 = pd.read_csv(f'{path1}')
    df2 = pd.read_csv(f'{path2}')
    dataset1 = df1['sentences'].tolist()
    dataset2 = df2['sentences'].tolist()
    return dataset1, dataset2


def word_alignment(dataset1, dataset2):
    alignment_list = []
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="i")

    # input texts should be tokenized
    for i in tqdm(range(len(dataset1))):
        alignments = myaligner.get_word_aligns(word_tokenize(dataset1[i]), word_tokenize(dataset2[i]))
        alignment_list.append(alignments['itermax'])
    return alignment_list


def main(args):
    path = f'{args.output}/augmentation/word-alignment'
    if not os.path.isdir(path): os.makedirs(path)
    dataset1, dataset2 = load(args.data1, args.data2)
    alignments = word_alignment(dataset1, dataset2)
    a_df = pd.DataFrame(columns=['alignments'])
    for i in range(len(alignments)):
        a_df.at[i, 'alignments'] = alignments[i]
    data1_name = str(args.data1)[str(args.data1).rindex('/') + 1:str(args.data1).rindex('.')]
    data2_name = str(args.data2)[str(args.data2).rindex('/') + 1:str(args.data2).rindex('.')]
    a_df.to_csv(f'{path}/{data1_name}.{data2_name}.alignments.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word Alignment')
    parser.add_argument('--data1', dest='data1', type=str, default='../../output/back-translation/word-alignment/D.csv',
                        help='First dataset1 file path')
    parser.add_argument('--data2', dest='data2', type=str,
                        default='../../output/back-translation/word-alignment/D_deu.csv',
                        help='Second dataset file path')
    parser.add_argument('--output', dest='output', type=str, default='../../output',
                        help='output path, e.g., ../output')
    args = parser.parse_args()

    main(args)
