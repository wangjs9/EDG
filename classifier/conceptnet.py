import requests

class conceptNet:
    def __init__(self):
        self.url = 'http://api.conceptnet.io/'

    def lookup(self, word):
        url_to_search = self.url + 'c/en/' + word
        WordInfo = requests.get(url_to_search).json()
        return WordInfo

    def relation(self, rel, concept):
        url_to_search = self.url + 'search?rel=/r/' + rel + "&end=/c/en/" + concept
        RelInfo = requests.get(url_to_search).json()
        return RelInfo

    def termsAsscociation(self, origin, destiny, limit):
        url_to_search = self.url + 'assoc/list/en/' + origin + ',' + destiny + '@-1?limit=' + str(
            limit) + '&filter=/c/en'
        AssoInfo = requests.get(url_to_search).json()
        return AssoInfo

    def relatedTerms(self, word):
        word_info = self.lookup(word)
        edges = word_info['edges']
        nodes = [word]
        # nodes, weights = list(), list()

        for idx, edge in enumerate(edges):
            node = edge['start']
            if node['language'] == 'en'and node['label'] not in nodes:
                nodes.append(node['label'])
                # weights.append(node['weight'])
            node = edge['end']
            if node['language'] == 'en' and node['label'] not in nodes:
                nodes.append(node['label'])

        return nodes[1:]
        # return nodes, weights

