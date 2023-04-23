import os, spacy
from tqdm import tqdm
import xml.etree.ElementTree as et

#nlp = spacy.load("en_core_web_sm")  # en_core_web_trf for transformer-based; error ==> python -m spacy download en_core_web_sm

from cmn.review import Review

class SemEvalReview(Review):

    def __init__(self, id, sentences, time, author, aos):
        super().__init__(self, id, sentences, time, author, aos)

    @staticmethod
    def txtloader(path):
        reviews = []
        with tqdm(total=os.path.getsize(path)) as pbar, open(path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                pbar.update(len(line))
                sentence, aos = line.split('####')
                aos = aos.replace('\'POS\'', '+1').replace('\'NEG\'', '-1').replace('\'NEU\'', '0')

                # for the current datafile, each row is a review of single sentence!
                # sentence = nlp(sentence)
                reviews.append(Review(id=i, sentences=[[str(t).lower() for t in sentence.split()]], time=None, author=None,
                                      aos=[eval(aos)], lempos=None,
                                      parent=None, lang='eng_Latn'))
        return reviews

    @staticmethod
    def xmlloader(path):
        reviews_list = []
        xtree = et.parse(path).getroot()
        reviews = [SemEvalReview.parse(xsentence) for xreview in tqdm(xtree) for xsentences in xreview for xsentence in xsentences]
        return [r for r in reviews if r]

    @staticmethod
    def map_idx(aspect, text):
        # aspect: ('token', from_char, to_char)
        text_tokens = text[:aspect[1]].split()
        aspect_tokens = aspect[0].split()

        tmp = [*text] #mutable string :)
        tmp[aspect[1]: aspect[2]] = [' '] + [*aspect[0]] + [' ']
        text = ''.join(tmp)

        return [i for i in range(len(text_tokens), len(text_tokens) + len(aspect_tokens))], text

    @staticmethod
    def parse(xsentence):
        id = xsentence.attrib["id"]
        aos = []
        has_opinion = False
        for element in xsentence:
            if element.tag == 'text': sentence = element.text # we consider each sentence as a signle review
            if element.tag == "Opinions":
                #<Opinion target="place" category="RESTAURANT#GENERAL" polarity="positive" from="5" to="10"/>
                for opinion in element:
                    if opinion.attrib["target"] == 'NULL': continue
                    # we may have duplicates for the same aspect due to being in different category like in semeval 2016's <sentence id="1064477:4">
                    aspect = (opinion.attrib["target"], int(opinion.attrib["from"]), int(opinion.attrib["to"])) #('place', 5, 10)
                    # we need to map char index to token index in aspect
                    aspect, sentence = SemEvalReview.map_idx(aspect, sentence)
                    category = opinion.attrib["category"] # 'RESTAURANT#GENERAL'
                    sentiment = opinion.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0') #'+1'
                    aos.append((aspect, [], sentiment))
                aos = sorted(aos, key=lambda x: int(x[0][0])) #based on start of sentence
        #sentence = nlp(sentence) # as it does some processing, it destroys the token idx for aspect term
        return Review(id=id, sentences=[[str(t).lower() for t in sentence.split()]], time=None, author=None,
                      aos=aos, lempos=None,
                      parent=None, lang='eng_Latn') if aos else None

    def xmlloader2014(path):
        reviews_list = []
        tree = et.parse(path)
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
