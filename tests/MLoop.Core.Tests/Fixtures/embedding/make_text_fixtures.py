"""Writes the text-encoder fixtures the embedding tests read.

    python make_text_fixtures.py   (needs numpy, onnx, sentencepiece, transformers)

- tiny-tokenizer.model: a SentencePiece Unigram model (NFKC, as XLM-RoBERTa's) trained on a few Korean and English
  sentences — small enough to commit, a real model proto all the same.
- tiny-text-encoder.onnx: input_ids, attention_mask, token_type_ids [N,T] (int64) -> one row of E per id ->
  [N,T,4] "last_hidden_state", shaped like a sentence encoder's output so the mean-of-tokens path is the one
  exercised. attention_mask and token_type_ids are inputs it does not read, as an exported BERT may not.
- tiny-text-encoder-no-types.onnx: the same without a token_type_ids input (an XLM-RoBERTa export has none).
- text.json: texts; their ids as transformers' XLMRobertaTokenizer numbers them with this model (fairseq: <s> 0,
  <pad> 1, </s> 2, <unk> 3, every other piece its SentencePiece number + 1), cut at MAX_TOKENS; and the L2-normalised
  mean of E over those ids.
"""
import json
import os
import tempfile

import numpy as np
import onnx
import sentencepiece as spm
from onnx import TensorProto, helper
from transformers import XLMRobertaTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOKENS = 12
DIM = 4

CORPUS = [
    "배송이 빨라서 좋았어요", "배송이 늦어서 별로였어요", "상품이 좋아요 다시 살게요", "포장이 꼼꼼해요",
    "the delivery was fast", "the product is good", "I would buy it again", "packing was careful",
    "가격 대비 만족합니다", "색상이 사진과 달라요", "quality is fine for the price", "colour differs from the photo",
] * 4

with tempfile.TemporaryDirectory() as tmp:
    corpus = os.path.join(tmp, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(CORPUS))
    prefix = os.path.join(tmp, "tiny")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, vocab_size=80, model_type="unigram",
        normalization_rule_name="nmt_nfkc", character_coverage=1.0, seed_sentencepiece_size=10000,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, num_threads=1)
    with open(prefix + ".model", "rb") as src, open(os.path.join(HERE, "tiny-tokenizer.model"), "wb") as dst:
        dst.write(src.read())

tokenizer = XLMRobertaTokenizer(vocab_file=os.path.join(HERE, "tiny-tokenizer.model"))
vocab = len(tokenizer)
rng = np.random.default_rng(11)
E = rng.normal(size=(vocab, DIM)).astype(np.float32)

TEXTS = [
    "배송이 빨라서 좋았어요",
    "the product is good",
    "ＡＢＣ 전각 ①",              # NFKC folds these before cutting
    "  여러   칸의  공백 ",          # runs of white space are one
    "처음 보는 글자 魑魅 😀",        # pieces the model never saw -> <unk>
    "배송이 빨라서 좋았어요 상품이 좋아요 다시 살게요 포장이 꼼꼼해요 가격 대비 만족합니다",  # longer than MAX_TOKENS
    "",
]

cases = []
for text in TEXTS:
    ids = tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    full = tokenizer(text)["input_ids"]
    vector = E[ids].mean(axis=0)
    vector = vector / np.linalg.norm(vector)
    cases.append({"text": text, "ids": ids, "truncated": len(full) > MAX_TOKENS, "vector": vector.tolist()})

with open(os.path.join(HERE, "text.json"), "w", encoding="utf-8") as f:
    json.dump({"maxTokens": MAX_TOKENS, "dimension": DIM, "cases": cases}, f, ensure_ascii=False, indent=1)


def encoder(name, with_types):
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["N", "T"]),
              helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["N", "T"])]
    if with_types:
        inputs.append(helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, ["N", "T"]))
    graph = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["last_hidden_state"], axis=0)],
        name, inputs,
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["N", "T", DIM])],
        [helper.make_tensor("E", TensorProto.FLOAT, [vocab, DIM], E.flatten().tolist())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="mloop-fixture")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(HERE, name + ".onnx"))


encoder("tiny-text-encoder", with_types=True)
encoder("tiny-text-encoder-no-types", with_types=False)
