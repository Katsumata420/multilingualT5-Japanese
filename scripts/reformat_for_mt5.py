import pandas as pd
import sys
import os

assert len(sys.argv) == 3

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_json(input_file, orient='records', lines=True)
df['inputs'] = df['src'].str.replace('\n', '').str.replace('\t', '')
df['targets'] = df['tgt'].str.replace('\n', '').str.replace('\t', '')

df[['inputs', 'targets']].to_csv(output_file, sep='\t', index=False)
