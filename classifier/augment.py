from senticnet6 import senticnet
from conceptnet import conceptNet

CN = conceptNet()
words = CN.relatedTerms('abandoned')
print(words)
