from cmn.review import Review


class TwitterReview(Review):
    def __init__(self, id, sentences, time, author, aos):
        super().__init__(self, id, sentences, time, author, aos)

    def load(path):
        return TwitterReview._loader(path)

    @staticmethod
    def _loader(path):
        reviews_list = []
        r_id = 0
        with open(path, 'r', encoding="utf8") as f:
            lines = [line.strip() for line in f]

        for idx in range(0, len(lines), 3):
            text, aspect, sentiment = lines[idx:idx + 3]
            current_text = text.replace('$T$', aspect)
            aos_list_list = [[(list(range(text.split().index('$T$'),
                                          text.split().index('$T$') + len(aspect.split()))), [], sentiment)]]
            reviews_list.append(
                Review(id=r_id, sentences=[[str(t).lower() for t in current_text.split()]], time=None,
                       author=None, aos=aos_list_list, lempos=""))
            r_id += 1
        return reviews_list
