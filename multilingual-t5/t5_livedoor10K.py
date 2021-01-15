import t5.data
from t5.data import sentencepiece_vocabulary
from t5.evaluation import metrics
from t5.data import preprocessors
from t5.data import TaskRegistry
from t5.data import TextLineTask

import numpy as np
import functools
import tensorflow as tf
from sumeval.metrics.rouge import RougeCalculator

rouge_cal = RougeCalculator(stopwords=True, lang="ja")

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
DEFAULT_SPM_PATH = "/home/katsumata/work/summarization/mt5/sentencepiece.model"
DEFAULT_VOCAB = sentencepiece_vocabulary.SentencePieceVocabulary(
    DEFAULT_SPM_PATH)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}

# rouge-1, rouge-2, rouge-lを評価指標とします
def rouge(targets, predictions):
  predictions = [tf.compat.as_text(x) for x in predictions]

  if isinstance(targets[0], list):
    targets = [[tf.compat.as_text(x) for x in target] for target in targets]
  else:
    targets = [tf.compat.as_text(x) for x in targets]
    targets = [targets]

  list_1, list_2, list_l = [], [], []
  for i in range(len(predictions)):
    list_1.append(rouge_cal.rouge_n(
            summary=predictions[i],
            references=targets[0][i],
            n=1))
    list_2.append(rouge_cal.rouge_n(
            summary=predictions[i],
            references=targets[0][i],
            n=2))
    list_l.append(rouge_cal.rouge_l(
            summary=predictions[i],
            references=targets[0][i]))

  return {"rouge_1": np.array(list_1).mean(),
          "rouge_2": np.array(list_2).mean(),
          "rouge_l": np.array(list_l).mean()}

task_name = "t5_livedoor10K"

tsv_path = {
    "train": "/home/katsumata/work/summarization/mt5/data/livedoor_10K/train.tsv",
    "validation": "/home/katsumata/work/summarization/mt5/data/livedoor_10K/dev.tsv",
    "test": "/home/katsumata/work/summarization/mt5/data/livedoor_10K/test.tsv",
}

TaskRegistry.add(
    task_name,
    TextLineTask,
    split_to_filepattern=tsv_path,
    text_preprocessor=[
      functools.partial(
          preprocessors.parse_tsv,
          field_names=["inputs", "targets"]),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[rouge])
