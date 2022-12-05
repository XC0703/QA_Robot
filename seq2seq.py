# -*- coding: utf-8 -*-

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable Tensorflow debug message

import jieba
import tensorflow as tf
from tqdm import tqdm

from data_processing import Data, add_flag


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)

    def call(self, X):
        X = self.embedding(X)
        output, state = self.gru(X)
        return output, state


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, X, state, **kwargs):
        X = self.embedding(X)
        context = tf.reshape(tf.repeat(state, repeats=X.shape[1], axis=0), (X.shape[0], X.shape[1], -1))
        X_and_context = tf.concat((X, context), axis=2)
        output, state = self.gru(X_and_context)
        output = tf.reshape(output, (-1, output.shape[2]))
        X = self.fc(output)
        return X, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))


class Seq2Seq(object):
    def __init__(self, config):
        self.config = config
        vacab_size_in = config["vacab_size_in"]
        vacab_size_out = config["vacab_size_out"]
        embedding_dim = config["embedding_dim"]
        self.units = config["layer_size"]
        self.batch_size = config["batch_size"]
        self.encoder = Encoder(vacab_size_in, embedding_dim, self.units, self.batch_size)
        self.decoder = Decoder(vacab_size_out, embedding_dim, self.units, self.batch_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.ckpt_dir = self.config["model_data"]
        logging.basicConfig(level=logging.INFO)
        self.LOG = logging.getLogger("Seq2Seq")
        if tf.io.gfile.listdir(self.ckpt_dir):
            self.LOG.info("正在加载模型")
            self.checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_dir))

        data = Data(config)
        self.dataset, self.tokenizer_in, self.tokenizer_out = data.load()

    def loss_function(self, real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def training_step(self, src, tgt, tgt_lang):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(src)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([tgt_lang.word_index["bos"]] * self.batch_size, 1)
            for t in range(1, tgt.shape[1]):
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
                loss += self.loss_function(tgt[:, t], predictions)
                dec_input = tf.expand_dims(tgt[:, t], 1)
        step_loss = (loss / int(tgt.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return step_loss

    def train(self):
        writer = tf.summary.create_file_writer(self.config["log_dir"])
        self.LOG.info(f"数据目录: {self.config['data_path']}")

        epoch = 0
        train_epoch = self.config["epochs"]
        while epoch < train_epoch:
            total_loss = 0
            iter_data = tqdm(self.dataset)
            for batch, (src, tgt) in enumerate(iter_data):
                batch_loss = self.training_step(src, tgt, self.tokenizer_out)
                total_loss += batch_loss
                iter_data.set_postfix_str(f"batch_loss: {batch_loss:.4f}")

            self.checkpoint.save(file_prefix=os.path.join(self.ckpt_dir, "ckpt"))
            epoch = epoch + 1

            self.LOG.info(f"Epoch: {epoch}/{train_epoch} Loss: {total_loss:.4f}")
            with writer.as_default():
                tf.summary.scalar("loss", total_loss, step=epoch)

    def predict(self, sentence):
        max_length = self.config["max_length"]
        sentence = " ".join(jieba.cut(sentence))
        sentence = add_flag(sentence)

        inputs = self.tokenizer_in.texts_to_sequences([sentence])
        inputs = [[x for x in inputs[0] if x if not None]]  # Remove None. TODO: Why there're None???

        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length, padding="post")
        inputs = tf.convert_to_tensor(inputs)

        enc_out, enc_hidden = self.encoder(inputs)
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self.tokenizer_out.word_index["bos"]], 0)
        result = ""
        for _ in range(max_length):
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
            if self.tokenizer_out.index_word[predicted_id] == "eos":
                break

            result += str(self.tokenizer_out.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)

        return result
