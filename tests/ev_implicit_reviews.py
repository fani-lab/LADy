"""Expected values for test cases of implicit/explicit review loading"""

# Expected first, last, and amount of reviews with implicit aspects"""
SEMEVAL_IMPLICIT = {
    "first": {
        'id': '1004293:2',
        'text': 'they never brought us complimentary noodles, ignored repeated requests for sugar, and threw our dishes on the table.',
        'sentences': [
            ['they', 'never', 'brought', 'us', 'complimentary', 'noodles,', 'ignored', 'repeated', 'requests', 'for', 'sugar,', 'and', 'threw', 'our', 'dishes', 'on', 'the', 'table.']
        ],
        'aos': [[([None], [], '-1')]],
        'lang': 'eng_Latn',
        'orig': True
    },
    "last": {
        'id': '1058221:7',
        'text': 'the last time i walked by it looked pretty empty. hmmm.', 
        'sentences': [
            ['the', 'last', 'time', 'i', 'walked', 'by', 'it', 'looked', 'pretty', 'empty.', 'hmmm.']
        ], 
        'aos': [[([None], [], '-1')]], 
        'lang': 'eng_Latn', 
        'orig': True
    },
    "count": 10,
}

# Expected first, last, and amount of reviews with explicit aspects
SEMEVAL_EXPLICIT = {
    "first": {
        'id': '1004293:0',
        'text': 'judging from previous posts of test this used to be a good place but not any longer.',
        'sentences': [
            ['judging', 'from', 'previous', 'posts', 'of', 'test', 'this', 'used', 'to', 'be', 'a', 'good', 'place', 'but', 'not', 'any', 'longer.']
        ],
        'aos': [
            [(['posts', 'of', 'test'], [], '-1'),
             (['place'], [], '-1')]
        ],
        'lang': 'eng_Latn',
        'orig': True
    },
    "last": {
        'id': '1058221:4',
        'text': 'i happen to have a policy that goes along with a little bit of self-respect, which includes not letting a waiter intimidate me, i.e. make me feel bad asking for trivialities like water, or the check.',
        'sentences': [
            ['i', 'happen', 'to', 'have', 'a', 'policy', 'that', 'goes', 'along', 'with', 'a', 'little', 'bit', 'of', 'self-respect,', 'which', 'includes', 'not', 'letting', 'a', 'waiter', 'intimidate', 'me,', 'i.e.', 'make', 'me', 'feel', 'bad', 'asking', 'for', 'trivialities', 'like', 'water,', 'or', 'the', 'check.']
        ],
        'aos': [[(['waiter'], [], '-1')]],
        'lang': 'eng_Latn',
        'orig': True
    },
    "count": 18,
}

# Expected first, last, and amount of reviews with implicit and explicit aspects"""
SEMEVAL_BOTH = {
    "first": {
        'id': '1004293:0',
        'text': 'judging from previous posts of test this used to be a good place but not any longer.',
        'sentences': [
            ['judging', 'from', 'previous', 'posts', 'of', 'test', 'this', 'used', 'to', 'be', 'a', 'good', 'place', 'but', 'not', 'any', 'longer.']
        ],
        'aos': [
            [(['posts', 'of', 'test'], [], '-1'),
             (['place'], [], '-1')]
        ],
        'lang': 'eng_Latn',
        'orig': True
    },
    "last": {
        'id': '1058221:7',
        'text': 'the last time i walked by it looked pretty empty. hmmm.', 
        'sentences': [
            ['the', 'last', 'time', 'i', 'walked', 'by', 'it', 'looked', 'pretty', 'empty.', 'hmmm.']
        ], 
        'aos': [[([None], [], '-1')]], 
        'lang': 'eng_Latn', 
        'orig': True
    },
    "count": 26,
}

SEMEVAL_NULL = {
    "count": 0,
}
