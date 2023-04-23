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
        nlp = spacy.load("en_core_web_sm")  # en_core_web_trf for transformer-based; error ==> python -m spacy download en_core_web_sm
        with tqdm(total=os.path.getsize(path)) as pbar, open(path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                pbar.update(len(line))
                sentences, aos = line.split('####')
                aos = aos.replace('\'POS\'', '+1').replace('\'NEG\'', '-1').replace('\'NEU\'', '0')

                # for the current datafile, each row is a review of single sentence!
                sentences = nlp(sentences)
                reviews.append(Review(id=i, sentences=[[str(t).lower() for t in sentences]], time=None, author=None,
                                      aos=[eval(aos)], lempos=[[(t.lemma_.lower(), t.pos_) for t in sentences]],
                                      parent=None, lang='eng_Latn'))
        return reviews

    def xmlloader(path):
        reviews_list = []
        nlp = spacy.load("en_core_web_sm")
        tree = ET.parse(path)
        reviews = tree.getroot()
        for review in reviews:
            for sentences in review:
                for sentence in sentences:
                    sid = sentence.attrib["id"]
                    review_list = []
                    aos_list = []
                    text = ""
                    tokens = []
                    has_opinion = False
                    for data in sentence:

                        if data.tag == 'text':
                            text = data.text
                            review_list.append(text)

                        if data.tag == "Opinions":
                            has_opinion = True
                            current_text = review_list.pop()
                            opinion_list = [o for o in data if o.attrib["target"] != "NULL"]
                            opinion_list = sorted(opinion_list, key=lambda x: int(x.attrib["from"]))
                            alt_opinion_list = []
                            for op in opinion_list:
                                if alt_opinion_list and (op.attrib["target"], op.attrib["from"], op.attrib["to"]) == (alt_opinion_list[-1].attrib["target"], alt_opinion_list[-1].attrib["from"], alt_opinion_list[-1].attrib["to"]):
                                    continue
                                alt_opinion_list.append(op)
                            opinion_list = alt_opinion_list
                            idx_tuples = []
                            # null_count = 0
                            for j, opinion in enumerate(opinion_list):
                                aspect = opinion.attrib["target"]
                                aspect_list = aspect.split()
                                if len(aspect_list) == 0:  # if aspect is NULL
                                    continue
                                letter_index_tuple = (int(opinion.attrib['from']) + 2 * j,
                                                      int(opinion.attrib['to']) + 2 * j)

                                current_text = current_text[
                                               0:letter_index_tuple[0]] + ' ' + aspect + ' ' + current_text[
                                                                                               letter_index_tuple[1]:]
                                idx_tuples.append((letter_index_tuple[0] + 1, letter_index_tuple[1] + 1))
                            review_list.append(current_text)
                            tokens = current_text.split()
                            for j, opinion in enumerate(opinion_list):
                                aspect = opinion.attrib["target"]
                                aspect_list = aspect.split()
                                # if text == "I've enjoyed 99% of the dishes we've ordered with the only exceptions being the occasional too-authentic-for-me dish (I'm a daring eater but not THAT daring).":
                                #     break
                                if len(aspect_list) == 0:  # if aspect is NULL
                                    continue
                                letter_index_tuple = idx_tuples[j]
                                sentiment = opinion.attrib["polarity"].replace('positive', '+1').replace('negative',
                                                                                                         '-1').replace(
                                    'neutral', '0')
                                idx_of_from = [i for i in range(len(current_text)) if
                                               current_text.startswith(aspect, i)].index(letter_index_tuple[0])

                                idx_start_token_of_aspect = [i for i in range(len(tokens)) if
                                                             i + len(aspect_list) <= len(
                                                            tokens) and SemEvalReview.compare_tokens(
                                                            tokens[i:i + len(aspect_list)], aspect_list)][idx_of_from]

                                idx_aspect_list = list(
                                    range(idx_start_token_of_aspect, idx_start_token_of_aspect + len(aspect_list)))
                                aos = (idx_aspect_list, [], eval(sentiment))
                                if len(aos) != 0:
                                    aos_list.append(aos)
                            if len(aos_list) == 0:  # if all aspects were NULL, we remove sentence
                                review_list.pop()
                                break
                    if not has_opinion:  # if sentence did not have any opinion, we remove it
                        review_list.pop()
                    # if len(aos_list_list) != 0:
                    #     aos_list_list.append(aos_list)
                    if len(aos_list) == 0 or len(review_list) == 0:
                        continue

                    reviews_list.append(
                        Review(id=sid, sentences=[[str(t).lower() for t in s.split()] for s in review_list], time=None,
                               author=None, aos=[aos_list], lempos=""))
        return reviews_list

    @staticmethod
    def compare_tokens(tokens_list, aspect_list):
        for t, a in zip(tokens_list, aspect_list):
            if a not in t:
                return False
        return True

    def xmlloader2014(path):
        reviews_list = []
        tree = ET.parse(path)
        sentences = tree.getroot()
        for sentence in sentences:
            i = sentence.attrib["id"]
            review_list = []
            # aos_list_list = []
            # for sentence in sentences:
            # for sentence in sentences:
            aos_list = []
            text = ""
            tokens = []
            has_opinion = False
            for data in sentence:

                if data.tag == 'text':
                    text = data.text
                    review_list.append(text)

                if data.tag == "aspectTerms":
                    has_opinion = True
                    current_text = review_list.pop()
                    previous_idx_tuple = None
                    aspectTerm_list = [o for o in data]
                    aspectTerm_list = sorted(aspectTerm_list, key=lambda x: int(x.attrib["from"]))
                    idx_tuples = []
                    for j, aspectTerm in enumerate(aspectTerm_list):
                        aspect = aspectTerm.attrib["term"]
                        aspect_list = aspect.split()
                        if aspect == "NULL" or len(aspect_list) == 0:  # if aspect is NULL
                            continue
                        letter_index_tuple = (
                        int(aspectTerm.attrib['from']) + 2 * j, int(aspectTerm.attrib['to']) + 2 * j)
                        # current_text = current_text.replace('  ', ' ')

                        current_text = current_text[
                                       0:letter_index_tuple[0]] + ' ' + aspect + ' ' + current_text[
                                                                                       letter_index_tuple[1]:]
                        idx_tuples.append((letter_index_tuple[0] + 1, letter_index_tuple[1] + 1))
                    review_list.append(current_text)
                    tokens = current_text.split()
                    for j, aspectTerm in enumerate(aspectTerm_list):
                        aspect = aspectTerm.attrib["term"]
                        letter_index_tuple = idx_tuples[j]
                        aspect_list = aspect.split()
                        if aspect == "NULL" or len(aspect_list) == 0:  # if aspect is NULL
                            continue
                        sentiment = aspectTerm.attrib["polarity"].replace('positive', '+1').replace('negative',
                                                                                                    '-1').replace(
                            'conflict', '0').replace('neutral', '0')
                        idx_of_from = [i for i in range(len(current_text)) if
                                       current_text.startswith(aspect, i)].index(letter_index_tuple[0])

                        # print(i, flush=True)
                        idx_start_token_of_aspect = [i for i in range(len(tokens)) if i + len(aspect_list) <= len(
                            tokens) and SemEvalReview.compare_tokens(tokens[i:i + len(aspect_list)], aspect_list)][
                            idx_of_from]
                        idx_aspect_list = list(
                            range(idx_start_token_of_aspect, idx_start_token_of_aspect + len(aspect_list)))
                        aos = (idx_aspect_list, [], eval(sentiment))
                        if len(aos) != 0:
                            aos_list.append(aos)
                    if len(aos_list) == 0:  # if all aspects were NULL, we remove sentence
                        review_list.pop()
                        break
            if not has_opinion:  # if sentence did not have any opinion, we remove it
                review_list.pop()
            reviews_list.append(
                Review(id=i, sentences=[[str(t).lower() for t in s.split()] for s in review_list], time=None,
                       author=None, aos=[aos_list], lempos=""))
        return reviews_list
# if __name__ == '__main__':
#     reviews = SemEvalReview.load(r'C:\Users\Administrator\Github\Fani-Lab\pxp-topicmodeling-working\data\raw\semeval-umass\sam_eval2016.txt', None, None)
#     print(reviews[0].get_aos())
#     print(Review.to_df(reviews))
