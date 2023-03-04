import argparse, os, pickle
from time import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import sys
import pandas as pd
from tqdm import tqdm
# sys.path.insert(1, 'src/cmn')
# import review

# from src.cmn.review import Review


def load(input, output):
    df = pd.read_csv(f'{input}')
    corpus = df['sentences'].tolist()
    return corpus


def translate(model, tokenizer, corpus, source_lang, target_lang):
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=source_lang,
                          tgt_lang=target_lang, max_length=400)
    back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=target_lang,
                               tgt_lang=source_lang, max_length=400)
    translated = []
    back_translated = []
    for c in tqdm(corpus):
        # print(c)
        translated_text = translator(c)[0]["translation_text"]
        translated.append(translated_text)
        # print("Translated:\t", output[0]["translation_text"])
        back_translated_text = back_translator(translated_text)[0]["translation_text"]
        back_translated.append(back_translated_text)

        # translated_text.append(translator(c)[0]["translation_text"])
        # back_translated_text.append(back_translator(translated_text)[0]["translation_text"])
    return translated, back_translated


def main(args):
    path = f'{args.output}/augmentation/back-translation'
    if not os.path.isdir(path): os.makedirs(path)
    corpus = load(args.data, args.output)

    checkpoint = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Review.save_sentences(reviews, path)

    for m in args.mt:
        if m == "nllb":
            for language in args.lan:
                target_lang = language
                source_lang = "eng_Latn"
                translated_list, back_translated_list = translate(model, tokenizer, corpus, source_lang, target_lang)
                t_df = pd.DataFrame(translated_list, columns=["sentences"])
                bt_df = pd.DataFrame(back_translated_list, columns=["sentences"])
                lang = language[0:str(language).index('_')]
                t_df.to_csv(f'{path}/D.{lang}.csv', index=False)
                bt_df.to_csv(f'{path}/D_{lang}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Back-translation')
    parser.add_argument('--mt',  nargs='+', type=str.lower, required=True, default=['nllb'], help='a list of translator models')
    parser.add_argument('--lan',  nargs='+', type=str, required=True, default=['deu_Latn'], help='a list of desired languages')
    parser.add_argument('--data', dest='data', type=str, default='../../data/raw/semeval/reviews.csv', help='dataset file path, e.g., ..data/raw/semeval/reviews.csv')
    parser.add_argument('--output', dest='output', type=str, default='../../output', help='output path, e.g., ../output')
    args = parser.parse_args()

    main(args)