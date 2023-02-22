import argparse, os
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine


def load(path1, path2):
    df1 = pd.read_csv(f'{path1}')
    df2 = pd.read_csv(f'{path2}')
    dataset1 = df1['sentences'].tolist()
    dataset2 = df2['sentences'].tolist()
    return dataset1, dataset2


def similarity(dataset1, dataset2):
    model = SentenceTransformer("johngiorgi/declutr-small")
    similarity_list = []
    for i in tqdm(range(len(dataset1))):
        embeddings = model.encode([dataset1[i], dataset2[i]])
        semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
        similarity_list.append(semantic_sim)
        # if semantic_sim < 0.7:
        #     print('Original:\t', dataset1[i])
        #     print('Back-translated:\t', dataset2[i])
        #     print(semantic_sim)
    return similarity_list


def main(args):
    path = f'{args.output}/augmentation/semantic-similarity'
    if not os.path.isdir(path): os.makedirs(path)
    dataset1, dataset2 = load(args.data1, args.data2)
    similarity_list = similarity(dataset1, dataset2)
    s_df = pd.DataFrame(similarity_list, columns=['similarities'])
    data1_name = str(args.data1)[str(args.data1).rindex('/') + 1:str(args.data1).rindex('.')]
    data2_name = str(args.data2)[str(args.data2).rindex('/') + 1:str(args.data2).rindex('.')]
    s_df.to_csv(f'{path}/{data1_name}.{data2_name}.similarities.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Comparison')
    parser.add_argument('--data1', dest='data1', type=str, default='../../output/augmentation/back-translation/D.csv',
                        help='Original dataset file path')
    parser.add_argument('--data2', dest='data2', type=str,
                        default='../../output/augmentation/back-translation/D_deu.csv',
                        help='Back-translated dataset file path')
    parser.add_argument('--output', dest='output', type=str, default='../../output',
                        help='output path, e.g., ../output')
    args = parser.parse_args()

    main(args)
