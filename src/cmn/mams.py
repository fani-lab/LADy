import os, spacy
from tqdm import tqdm
import xml.etree.ElementTree as ET

from cmn.review import Review

class MAMSReview(Review):
    def __init__(self, id, sentences, time, author, aos):
        super().__init__(self, id, sentences, time, author, aos)

    @staticmethod
    def xmlloader(path):
        reviews_list = []
        nlp = spacy.load("en_core_web_sm")
        tree = ET.parse(path)
        sentences = tree.getroot()
        i = -1
        for sentence in sentences: # each sentence is an individual review, unlike SemEval16
            i += 1

            text = ""
            tokens = []
            aos_list_list = []

            for data in sentence:
                if data.tag == "text": # clean the associated aspect tokens from punctuations
                    raw_text = data.text
                    current_text = raw_text
                    opinion_text = sentence.findall(".//aspectTerm")
                    for o in opinion_text:
                        aspect = o.attrib["term"]
                        aspect_list = aspect.split()
                        if len(aspect_list) == 0: # contains no aspect (mams dataset doesn't have NULL aspects)
                            continue
                        letter_index_tuple = (int(o.attrib['from']), int(o.attrib['to']))
                        current_text = current_text.replace('  ', ' ')
                        current_text = current_text[0:letter_index_tuple[0]] + ' ' + aspect + ' ' + current_text[letter_index_tuple[1]+1:]
                        #print("processing text:" + str(current_text))
                    tokens = current_text.split()
                
                if data.tag == "aspectTerms":
                    aos_list = []
                    for o in data: # each o is an aspectTerm

                        sentiment = o.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0')

                        aspect = o.attrib["term"]
                        aspect_list = aspect.split() # the aspect may consist more than 1 word
                        if len(aspect_list) == 0:
                            continue
                        
                        letter_index_tuple = (int(o.attrib['from']), int(o.attrib['to']))

                        # find the aspect instance of all text instances of the phrase
                        #print(tokens)

                        text_incidences = [i for i in range(len(raw_text)) 
                                            if raw_text.startswith(aspect, i) 
                                            and not raw_text[i-1].isalpha()
                                            and not raw_text[i+len(aspect)].isalpha()]
                        #print("text incidences: " + str(text_incidences))
                        idx_of_from = text_incidences.index(letter_index_tuple[0])
                        #print("index of from: " + str(idx_of_from))

                        # find the location of the aspect token
                        start_token_of_aspect = [i for i in range(len(tokens)) 
                                                if i + len(aspect_list) <= len(tokens) 
                                                and tokens[i:i + len(aspect_list)] == aspect_list]
        
                        #print("start token of aspect: " + str(start_token_of_aspect))

                        idx_start_token_of_aspect = start_token_of_aspect[idx_of_from]

                        idx_aspect_list = list(
                                            range(idx_start_token_of_aspect, idx_start_token_of_aspect + len(aspect_list)))
                                            
                        # compile the final aos 3-tuple for each aspect
                        aos = (idx_aspect_list, [], eval(sentiment))

                        if len(aos) != 0:
                            aos_list.append(aos)

                    if len(aos_list) != 0:
                        aos_list_list.append(aos_list)

            if len(aos_list_list) == 0:  # if no aspect in the sentence, it is not added
                continue 

            reviews_list.append(
                            Review(id=i, sentences=[[str(t).lower() for t in current_text.split()]], time=None,
                            author=None, aos=aos_list_list, lempos=""))

        return reviews_list