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
                        text_incidences = [i for i in range(len(raw_text)) if raw_text.startswith(aspect, i)]
                        idx_of_from = text_incidences.index(letter_index_tuple[0])

                        # find the location of the aspect token
                        start_token_of_aspect = [i for i in range(len(tokens)) 
                                                if i + len(aspect_list) <= len(tokens) 
                                                and tokens[i:i + len(aspect_list)] == aspect_list]
                        idx_start_token_of_aspect = start_token_of_aspect[idx_of_from]

                        idx_aspect_list = list(
                                            range(idx_start_token_of_aspect, idx_start_token_of_aspect + len(aspect_list)))
                                            
                        aos = (idx_aspect_list, [], eval(sentiment))

                        if len(aos) != 0:
                            aos_list.append(aos)

                    if len(aos_list) != 0:
                        aos_list_list.append(aos_list)

            if len(aos_list_list) == 0:  # if all aspects were NULL, we remove sentence
                continue 

            reviews_list.append(
                            Review(id=i, sentences=[[str(t).lower() for t in current_text.split()]], time=None,
                            author=None, aos=aos_list_list, lempos=""))

        return reviews_list

"""
{'id': 0, 'sentences': [['the', 'food', 'is', 'very', 'average', '...', 'the', 'thai', 'fusion', 'stuff', 'is', 'a', 'bit', 'too', 'sweet', ',', 'every', 'thing', 'they', 'serve', 'is', 'too', 'sweet', 'here', '.']], 
'aos': [[(['food'], ['average'], -1), (['thai', 'fusion', 'stuff'], ['too', 'sweet'], -1)]]}

[[
    (['food'], ['average'], -1), 
    (['thai', 'fusion', 'stuff'], ['too', 'sweet'], -1)
    ]]

<text>The food is very average...the Thai fusion stuff is a bit too sweet, every thing they serve is too sweet here.</text>
<Opinions>
    <Opinion target="food" category="FOOD#QUALITY" polarity="negative" from="4" to="8"/>
    <Opinion target="Thai fusion stuff" category="FOOD#QUALITY" polarity="negative" from="31" to="48"/>
    <Opinion target="NULL" category="FOOD#QUALITY" polarity="negative" from="0" to="0"/>
</Opinions>


        id                                          sentences                                                aos
0        0  [[the, food, is, very, average, ..., the, thai...  [[(['food'], ['average'], -1), (['thai', 'fusi...
1        1  [[we, went, around, 9:30, on, a, friday, and, ...                    [[(['service'], ['great'], 1)]]
2        2                    [[we, love, th, pink, pony, .]]                [[(['pink', 'pony'], ['love'], 1)]]
3        3                       [[the, food, is, decent, .]]                      [[(['food'], ['decent'], 0)]]
4        4  [[however, ,, it, 's, the, service, that, leav...            [[(['service'], ['bad', 'taste'], -1)]]
...    ...                                                ...                                                ...
1388  1388                                     [[bad, staff]]                       [[(['staff'], ['bad'], -1)]]
1389  1389             [[i, generally, like, this, place, .]]                       [[(['place'], ['like'], 1)]]
1390  1390                         [[the, food, is, good, .]]                        [[(['food'], ['good'], 1)]]
1391  1391       [[the, design, of, the, space, is, good, .]]                       [[(['space'], ['good'], 1)]]
1392  1392               [[but, the, service, is, horrid, !]]                  [[(['service'], ['horrid'], -1)]]

[1393 rows x 3 columns]
lda
"""