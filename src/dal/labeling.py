import argparse, os
import pandas as pd


def load(dataset_path, alignment_path, original_path, semantic_path):
    dataset_df = pd.read_csv(f'{dataset_path}')
    alignments_df = pd.read_csv(f'{alignment_path}')
    original_df = pd.read_csv(f'{original_path}')
    similarities_df = pd.read_csv(f'{semantic_path}')
    dataset = dataset_df['sentences'].tolist()
    alignments = alignments_df['alignments'].tolist()
    original_aos = original_df['aos'].tolist()
    similarities = similarities_df['similarities'].tolist()
    assert len(dataset) == len(alignments), "Back-translated dataset and alignments should have the same length"
    assert len(dataset) == len(original_aos), "Back-translated dataset and original dataset should have the same length"
    assert len(dataset) == len(similarities), "Back-translated dataset and similarities should have the same length"

    return dataset, alignments, original_aos, similarities


def labeling(dataset, alignments, original_aos, similarities):
    label_indices = []
    augmented_reviews = []
    augmented_reviews_labels = []
    for i in range(len(dataset)):
        if similarities[i] < 0.5:
            continue
        else:
            label_indices.append(i)
            augmented_reviews.append(dataset[i])
            labels_list_per_instance = []
            for aos_instance in eval(original_aos[i]):
                label_list_per_aspect = []
                for a in aos_instance[0]:
                    for alignment_instance in eval(alignments[i]):
                        if a == alignment_instance[0]:
                            label_list_per_aspect.append(alignment_instance[1])
                            break
                labels_list_per_instance.append(label_list_per_aspect)
            augmented_reviews_labels.append(labels_list_per_instance)
    return augmented_reviews, augmented_reviews_labels


def main(args):
    path = f'{args.output}/augmentation'
    if not os.path.isdir(path): os.makedirs(path)
    dataset, alignments, original_aos, similarities = load(args.dataset, args.alignment, args.original, args.semantic)
    augmented_reviews, augmented_reviews_labels = labeling(dataset, alignments, original_aos, similarities)
    a_df = pd.DataFrame(augmented_reviews, columns=['sentences'])
    a_df['labels'] = augmented_reviews_labels
    language = args.language.capitalize()

    a_df.to_csv(f'{path}/{language}.back-translated.with-labels.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word Alignment')
    parser.add_argument('--dataset', dest='dataset', type=str, default='../../output/augmentation/back-translation/D_deu.csv',
                        help='Dataset file path')
    parser.add_argument('--alignment', dest='alignment', type=str,
                        default='../../output/augmentation/word-alignment/D.D_deu.alignments.csv',
                        help='Alignment file path')
    parser.add_argument('--original', dest='original', type=str,
                        default='../../output/new/xml-2016/reviews_list.csv',
                        help='Original labels file path')
    parser.add_argument('--semantic', dest='semantic', type=str, default='../../output/augmentation/semantic-similarity/D.D_deu.similarities.csv',
                        help='Semantic file path')
    parser.add_argument('--lang', dest='language', type=str.lower, default='German',
                        help='Target languages, e.g., German')
    parser.add_argument('--output', dest='output', type=str, default='../../output',
                        help='output path, e.g., ../output')
    args = parser.parse_args()

    main(args)
