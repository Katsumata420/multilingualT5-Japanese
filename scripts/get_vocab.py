import sys

import sentencepiece as spm

model = sys.argv[1]
sp = spm.SentencePieceProcessor()
sp.Load(model)

print(sp.GetPieceSize())
