import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load(path):
    df = pd.read_csv(f'{path}')
    dataset = df['similarities'].tolist()
    return dataset

def load_2(path, path2):
    df = pd.read_csv(f'{path}')
    dataset = df['similarities'].tolist()

    df2 = pd.read_csv(f'{path2}')
    dataset2 = df2['similarities'].tolist()
    return dataset, dataset2


def semantic_similarity_histogram(dataset, language, output):
    fig, ax = plt.subplots()
    h = sns.histplot(data=dataset, color='purple', bins=10)
    x_range = [i / 10 for i in range(0, 11)]
    h.set(xticks=x_range)
    ax.set_ylim([0, 600])
    plt.title(language)
    plt.ylabel('Number of Reviews')
    plt.xlabel('Similarity Scores')
    plt.savefig(f'{output}{language}_Histogram_Plot.png')
    plt.clf()


def semantic_similarity_histogram_different_langs(dataset1, dataset2, language1, language2, output):
    df = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'value': dataset1, 'Language': language1}),
        pd.DataFrame.from_dict({'value': dataset2, 'Language': language2})
    ])

    h = sns.histplot(data=df, x='value', hue='Language', palette='magma', bins=10)
    x_range = [i / 10 for i in range(0, 11)]
    h.set(xticks=x_range)
    plt.title(f'{language1}_{language2}')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Similarity Scores')
    plt.savefig(f'{output}{language1}_{language2}_Histogram_Plot.png')
    plt.clf()


def main(args):
    # dataset_name = Path(args.input).stem

    dataset = load(args.input)
    language = args.language.capitalize()
    semantic_similarity_histogram(dataset, language, args.output)

    # dataset, dataset2 = load_2(args.input, args.input2)
    # language2 = args.language2.capitalize()
    # semantic_similarity_histogram_different_langs(dataset, dataset2, language, language2, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--input', dest='input', type=str,
                        default='../../output/augmentation/semantic-similarity/D.D_deu.similarities.csv',
                        help='Dataset file path')

    # parser.add_argument('--input2', dest='input2', type=str,
    #                     default='../../output/augmentation/semantic-similarity/D.D_arb.similarities.csv',
    #                     help='Dataset file path2')

    parser.add_argument('--output', dest='output', type=str, default='../../output/augmentation/semantic-similarity/',
                        help='output path, e.g., ../../output/augmentation/semantic-similarity/')

    parser.add_argument('--lang', dest='language', type=str.lower, default='German',
                        help='Target languages, e.g., German')

    # parser.add_argument('--lang2', dest='language2', type=str.lower, default='Arabic',
    #                     help='Target languages2, e.g., Arabic')

    args = parser.parse_args()

    main(args)
