import argparse, os
import pandas as pd


def load(datasets_path_list):
    df_list = []
    for p in datasets_path_list:
        df_list.append(pd.read_csv(p))
    return df_list


def merging(df_list):
    merged_df = pd.concat(df_list)
    return merged_df


def main(args):
    path = f'{args.output}/augmentation-R-16/augmented-with-labels'
    if not os.path.isdir(path): os.makedirs(path)
    df_list = load(args.datasets)
    merged_df = merging(df_list)
    merged_df.to_csv(f'{path}/All.back-translated.with-labels.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='joining')
    parser.add_argument('--datasets', nargs='+', type=str, help='a list of datasets paths')
    parser.add_argument('--output', dest='output', type=str, default='../../output',
                        help='output path, e.g., ../output')
    args = parser.parse_args()
    main(args)
