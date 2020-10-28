import pandas as pd
file = 'assertions.csv'
def ReadFile(FILE):
    data = pd.read_csv(FILE, delimiter='\t')
    data.columns = ['uri', 'relation', 'start', 'end', 'json']
    for index, row in data.iterrows():
        print(row)
        input()

ReadFile(file)