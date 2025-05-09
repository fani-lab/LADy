import os, spacy
from tqdm import tqdm
import xml.etree.ElementTree as ET

from cmn.review import Review

class MAMSReview(Review):
    def __init__(self, id, sentences, time, author, aos): super().__init__(self, id, sentences, time, author, aos)

    @staticmethod
    def load(path, explicit=False, implicit=True): return MAMSReview._xmlloader(path, explicit, implicit)
    
    @staticmethod
    def _xmlloader(path, explicit, implicit):
        reviews_list = []
        #nlp = spacy.load("en_core_web_sm") # spacy.load model not being used in code. Also throwing error
        tree = ET.parse(path)
        sentences = tree.getroot()
        i = -1
        for sentence in sentences:  # each sentence is an individual review, unlike SemEval16
            i += 1
            tokens = []
            aos_list_list = []
            raw_text = sentence.find("text").text
            current_text = raw_text
            aspect_terms = sentence.find("aspectTerms")

            # Handle explicit replacement
            if aspect_terms is not None:
                for o in aspect_terms:
                    aspect = o.attrib["term"]
                    if not implicit and aspect == 'NULL': continue
                    if not explicit and aspect != 'NULL': continue
                    aspect_list = aspect.split()
                    from_idx, to_idx = int(o.attrib['from']), int(o.attrib['to'])
                    current_text = current_text[:from_idx] + ' ' + aspect + ' ' + current_text[to_idx:]
                tokens = current_text.split()

            # Process aspects if tag exists
            if aspect_terms is not None:
                aos_list = []
                for o in aspect_terms:
                    aspect = o.attrib["term"]
                    if not implicit and aspect == 'NULL': continue
                    if not explicit and aspect != 'NULL': continue
                    sentiment = o.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0')
                    aspect_list = aspect.split()
                    from_idx = int(o.attrib['from'])
                    all_locs = [idx for idx in range(len(raw_text)) if raw_text.startswith(aspect, idx)]
                    try:
                        target_occurrence = all_locs.index(from_idx)
                        token_idx_list = [i for i in range(len(tokens)) if tokens[i:i+len(aspect_list)] == aspect_list]
                        idx_start = token_idx_list[target_occurrence]
                        aspect_indices = list(range(idx_start, idx_start + len(aspect_list)))
                        aos = (aspect_indices, [], eval(sentiment))
                        aos_list.append(aos)
                    except: continue

                if aos_list: aos_list_list.append(aos_list)
            implicit_arr = False
            # Implicit case when no aspectTerms tag or empty
            if implicit and (aspect_terms is None or not list(aspect_terms)): 
                aos_list_list.append([])
                implicit_arr = True

            if aos_list_list:
                reviews_list.append(
                    Review(id=i, sentences=[[str(t).lower() for t in current_text.split()]], 
                           time=None, author=None, lempos="", 
                           aos=aos_list_list, parent=None, lang='eng_Latn', implicit=implicit_arr))

        return reviews_list
