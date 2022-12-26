import os, spacy
from tqdm import tqdm
import xml.etree.ElementTree as ET

from cmn.review import Review

class SemEvalReview(Review):
    def __init__(self, id, sentences, time, author, aos):
        super().__init__(self, id, sentences, time, author, aos)

    @staticmethod
    def txtloader(path):
        reviews = []
        nlp = spacy.load("en_core_web_sm")  # en_core_web_trf for transformer-based
        with tqdm(total=os.path.getsize(path)) as pbar, open(path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                pbar.update(len(line))
                sentences, aos = line.split('####')
                aos = aos.replace('\'POS\'', '+1').replace('\'NEG\'', '-1').replace('\'NEU\'', '0')

                # for the current datafile, each row is a review of single sentence!
                sentences = nlp(sentences)
                reviews.append(Review(id=i, sentences=[[str(t).lower() for t in sentences]], time=None, author=None, aos=[eval(aos)], lempos=[[(t.lemma_.lower(), t.pos_) for t in sentences]]))
        return reviews

    def xmlloader(path):
        reviews_list = []
        nlp = spacy.load("en_core_web_sm")  # en_core_web_trf for transformer-based
        # ABSA16_Restaurants_Train_SB1_v2 True
        tree = ET.parse("ABSA16_Restaurants_Train_SB1_v2")
        reviews = tree.getroot()
        for review in reviews:
            i = review.attrib["rid"]
            sentences_list = []
            aos_list_list = []
            for sentences in review:
                for sentence in sentences:
                    aos_list = []
                    text = ""
                    modified_text = ""
                    tokens = []
                    for data in sentence:

                        if data.tag == 'text':
                            text = data.text
                            modified_text = text
                            for ch in "\'&;#$()*,-./:[]«»_!":
                                modified_text = modified_text.replace(ch, f" {ch}")
                            tokens = modified_text.split()
                            sentences_list.append(modified_text)

                        if data.tag == "Opinions":
                            aos = []  # then, it will be converted to tuple
                            polarity_list = []

                            for o in data:
                                aspect = o.attrib["target"]
                                aspect_list = aspect.split()
                                opinion = []
                                sentiment = o.attrib["polarity"].replace('\'negative\'', '+1').replace('\'polarity\'', '-1').replace('\'neutral\'', '0')
                                letter_index_tuple = (int(o.attrib['from']), int(o.attrib['to']))

                                idx_of_from = [i for i in range(len(text)) if
                                               text.startswith(aspect, i)].index(letter_index_tuple[0])

                                idx_start_token_of_aspect = [i for i in range(len(tokens)) if
                                                             i + len(aspect_list) < len(tokens) and tokens[i:i + len(
                                                                 aspect_list)] == aspect_list][idx_of_from]
                                idx_aspect_list = list(
                                    range(idx_start_token_of_aspect, idx_start_token_of_aspect + len(aspect_list)))
                                # print(tokens, idx_aspect)
                                aos = (idx_aspect_list, opinion, eval(sentiment))
                                aos_list.append(aos)
                    aos_list_list.append(aos_list)
                                # if aspect == "NULL":
                                #     continue
                                # opinion =
                                # aos.append()
            reviews.append(Review(id=i, sentences=[[str(t).lower() for t in nlp(s)] for s in sentences_list], time=None,
                                  author=None, aos=aos_list_list, lempos=[[(t.lemma_.lower(), t.pos_) for t in sentences]]))

            # for the current datafile, each row is a review of single sentence!
        return reviews


# if __name__ == '__main__':
#     reviews = SemEvalReview.load(r'C:\Users\Administrator\Github\Fani-Lab\pxp-topicmodeling-working\data\raw\semeval-umass\sam_eval2016.txt', None, None)
#     print(reviews[0].get_aos())
#     print(Review.to_df(reviews))