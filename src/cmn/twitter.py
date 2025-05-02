from cmn.review import Review


class TwitterReview(Review):
    def __init__(self, id, sentences, time, author, aos): super().__init__(self, id, sentences, time, author, aos)

    @staticmethod
    def load(path, explicit=True, implicit=False): return TwitterReview._loader(path, explicit, implicit)

    @staticmethod
    def _loader(path, explicit, implicit):
        reviews_list = []
        r_id = 0
        with open(path, 'r', encoding="utf8") as f: lines = [line.strip() for line in f]

        for idx in range(0, len(lines), 3):
            text, aspect, sentiment = lines[idx:idx + 3]
            
            if not implicit and aspect == 'NULL': continue
            if not explicit and aspect != 'NULL': continue
            
            current_text = text.replace('$T$', aspect if aspect != 'NULL' else '')
            implicit_arr = [aspect == 'NULL']

            if aspect != 'NULL': aos_list_list = [[(list(range(text.split().index('$T$'), text.split().index('$T$') + len(aspect.split()))), [], sentiment)]]
            else: aos_list_list = [[( [None], [], sentiment)]]               
            # print(aos_list_list)       
            reviews_list.append(Review(id=r_id, sentences=[[str(t).lower() for t in current_text.split()]], 
                                       time=None, author=None, lempos="", 
                                       aos=aos_list_list, implicit=implicit_arr))
            r_id += 1
        return reviews_list
