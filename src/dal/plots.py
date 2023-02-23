import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load(path):
    df = pd.read_csv(f'{path}')
    dataset = df['similarities'].tolist()
    return dataset


def semantic_similarity_histogram(dataset, language, output):
    h = sns.histplot(data=dataset, color='purple', bins=10)
    x_range = [i / 10 for i in range(0, 11)]
    h.set(xticks=x_range)
    plt.title(language)
    plt.ylabel('Number of Reviews')
    plt.xlabel('Similarity Scores')
    plt.savefig(f'{output}{language}_Histogram_Plot.png')
    plt.clf()


def main(args):
    dataset = load(args.input)
    # dataset_name = Path(args.input).stem
    language = args.language.capitalize()
    semantic_similarity_histogram(dataset, language, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--input', dest='input', type=str,
                        default='../../output/augmentation/semantic-similarity/D.D_deu.similarities.csv',
                        help='Dataset file path')
    parser.add_argument('--output', dest='output', type=str, default='../../output/augmentation/semantic-similarity/',
                        help='output path, e.g., ../../output/augmentation/semantic-similarity/')
    parser.add_argument('--lang', dest='language', type=str.lower, default='German',
                        help='Target languages, e.g., German')
    args = parser.parse_args()

    main(args)
