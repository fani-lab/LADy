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
        # ABSA16_Restaurants_Train_SB1_v2
        tree = ET.parse(path)
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
                    has_opinion = False
                    for data in sentence:

                        if data.tag == 'text':
                            text = data.text
                            modified_text = text
                            for ch in '&;#$()*,.[]«»_!()\':-\\/\"?%':
                                modified_text = modified_text.replace(ch, f" {ch} ")
                            tokens = modified_text.split()
                            # print("text", text)
                            # print("modified_text", modified_text)
                            sentences_list.append(modified_text)

                        if data.tag == "Opinions":
                            has_opinion = True
                            polarity_list = []
                            for o in data:
                                aspect = o.attrib["target"]
                                modified_aspect = aspect
                                for ch in '&;#$()*,.[]«»_!()\':-\\/\"?%':
                                    modified_aspect = modified_aspect.replace(ch, f" {ch} ")
                                aspect_list = modified_aspect.split()
                                if text == "I've enjoyed 99% of the dishes we've ordered with the only exceptions being the occasional too-authentic-for-me dish (I'm a daring eater but not THAT daring).":
                                    continue
                                if aspect == "NULL" or len(aspect_list) == 0:  # if aspect is NULL
                                    continue
                                # if len(aspect_list) == 0:  # if we do not have aspect
                                #     print("##############")
                                #     print(sentences_list.pop())
                                #     break

                                opinion = []
                                sentiment = o.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0')
                                letter_index_tuple = (int(o.attrib['from']), int(o.attrib['to']))
                                idx_of_from = [i for i in range(len(text)) if
                                               text.startswith(aspect, i)].index(letter_index_tuple[0])
                                idx_start_token_of_aspect = [i for i in range(len(tokens)) if
                                                             i + len(aspect_list) <= len(tokens) and tokens[i:i + len(
                                                                 aspect_list)] == aspect_list][idx_of_from]
                                idx_aspect_list = list(
                                    range(idx_start_token_of_aspect, idx_start_token_of_aspect + len(aspect_list)))

                                aos = (idx_aspect_list, opinion, eval(sentiment))
                                print("text", text)
                                print("aos", aos)
                                if len(aos) != 0:
                                    aos_list.append(aos)

                            if len(aos_list) == 0:  # if all aspects were NULL
                                sentences_list.pop()
                                break
                    if not has_opinion:  # sentence did not have any opinion, so we remove it
                        sentences_list.pop()
                    if len(aos_list) != 0:
                        aos_list_list.append(aos_list)
            reviews_list.append(Review(id=i, sentences=[[str(t).lower() for t in s.split()] for s in sentences_list], time=None,
                                  author=None, aos=aos_list_list, lempos=""))
        return reviews_list


# if __name__ == '__main__':
#     reviews = SemEvalReview.load(r'C:\Users\Administrator\Github\Fani-Lab\pxp-topicmodeling-working\data\raw\semeval-umass\sam_eval2016.txt', None, None)
#     print(reviews[0].get_aos())
#     print(Review.to_df(reviews))