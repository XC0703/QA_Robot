# -*- coding: utf-8 -*-

import io
import json
import logging
import os

import jieba
import tensorflow as tf
# import tensorflow_probability as tfp
from tqdm import tqdm


def add_flag(w):
    return "<bos> " + w + " <eos>"


class Data(object):
    def __init__(self, config) -> None:
        self.config = config
        self.seq_path = config["data_path"] + config["dataset"] + ".data"
        self.conv_path = config["data_path"] + config["dataset"] + ".conv"
        self.conv_size = os.path.getsize(self.conv_path)
        self.vacab_path_in = config["data_path"] + config["dataset"] + ".vin"
        self.vacab_path_out = config["data_path"] + config["dataset"] + ".vout"
        self.max_length = config["max_length"]
        self.batch_size = config["batch_size"]
        self.LOG = logging.getLogger("Data")
        logging.basicConfig(level=logging.INFO)
        jieba.setLogLevel(logging.INFO)  # Disable debug info

    def create_sequences(self):
        if os.path.exists(self.seq_path):  # Skip if processed data exists
            return

        if not os.path.exists(self.conv_path):
            self.LOG.info("找不到语料文件，请检查路径")
            exit()

        self.LOG.info("正在处理语料")
        with tqdm(total=self.conv_size) as pbar, open(self.conv_path, encoding="utf-8") as fin, open(self.seq_path, "w", encoding="utf-8") as fout:
            one_conv = ""  # 存储一次完整对话
            for line in fin:
                pbar.update(len(line.encode("utf-8")))
                line = line.strip("\n")
                if not line:  # Skip empty line
                    continue

                # Refer to dataset format: E M M
                if line[0] == self.config["e"]:  # E, end of conversation, save it
                    if one_conv:
                        fout.write(one_conv[:-1] + "\n")
                    one_conv = ""

                elif line[0] == self.config["m"]:  # M, question or answer, split them with \t
                    one_conv = one_conv + str(" ".join(jieba.cut(line.split(" ")[1]))) + "\t"

    def create_vacab(self, lang, vocab_path, vocab_size):
        if os.path.exists(vocab_path):  # Skip if exists
            return

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<UNK>")
        tokenizer.fit_on_texts(lang)
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json(ensure_ascii=False))

        self.LOG.info(f"正在保存: {vocab_path}")

    def create_vacabularies(self):
        if os.path.exists(self.vacab_path_in) and os.path.exists(self.vacab_path_out):  # Skip if exists
            return

        self.LOG.info(f"正在创建字典")
        lines = io.open(self.seq_path, encoding="UTF-8").readlines()
        word_pairs = [[add_flag(w) for w in l.split("\t")] for l in lines]
        input, target = zip(*word_pairs)
        self.create_vacab(input, self.vacab_path_in, self.config["vacab_size_in"])
        self.create_vacab(target, self.vacab_path_out, self.config["vacab_size_out"])

    def tokenize(self, path):
        # Load tokenizer from file
        with open(path, "r", encoding="utf-8") as f:
            tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
        return tokenizer

    def process(self):
        self.create_sequences()
        self.create_vacabularies()

    def load(self):
        self.process()  # Process dataset if not did before
        self.LOG.info("正在加载数据")
        lines = io.open(self.seq_path, encoding="UTF-8").readlines()
        word_pairs = [[add_flag(w) for w in l.split("\t")] for l in lines]
        words_in, words_out = zip(*word_pairs)
        tokenizer_in = self.tokenize(self.vacab_path_in)
        tokenizer_out = self.tokenize(self.vacab_path_out)

        tensor_in = tokenizer_in.texts_to_sequences(words_in)
        tensor_out = tokenizer_out.texts_to_sequences(words_out)
        tensor_in = tf.keras.preprocessing.sequence.pad_sequences(
            tensor_in, maxlen=self.max_length, truncating="post", padding="post")
        tensor_out = tf.keras.preprocessing.sequence.pad_sequences(
            tensor_out, maxlen=self.max_length, truncating="post", padding="post")

        self.steps_per_epoch = len(tensor_in) // self.batch_size
        BUFFER_SIZE = len(tensor_in)
        dataset = tf.data.Dataset.from_tensor_slices((tensor_in, tensor_out)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset, tokenizer_in, tokenizer_out
