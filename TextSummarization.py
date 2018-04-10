from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

import itertools

print('TensorFlow Version: {}'.format(tf.__version__))

reviews = pd.read_csv("./input/Reviews.csv")
reviews.head()
reviews.isnull().sum()
reviews = reviews.dropna()
reviews = reviews.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator',
                        'Score', 'Time'], 1)

reviews = reviews.reset_index(drop=True)
reviews.head()
for i in range(5):
    print("Review #", i + 1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=True):
    text = text.lower()
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

for i in range(5):
    print("Clean Review #", i + 1)
    print(clean_summaries[i])
    print(clean_texts[i])
    print()


def count_words(count_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


word_counts = {}
count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)

print("Size of Vocabulary:", len(word_counts))

embeddings_index = {}
with open('./input/numberbatch-en.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words / len(word_counts), 4) * 100

print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

vocab_to_int = {}
value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word
usage_ratio = round(len(vocab_to_int) / len(word_counts), 4) * 100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))

embedding_dim = 300
nb_words = len(vocab_to_int)
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

print('Size word embedding matrix: ', len(word_embedding_matrix))


def convert_to_ints(text, word_count, unk_count, eos=False):
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)
unk_percent = round(unk_count / word_count, 4) * 100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


def create_lengths(text):
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())


# print(np.percentile(lengths_texts.counts, 90))
# print(np.percentile(lengths_texts.counts, 95))
# print(np.percentile(lengths_texts.counts, 99))
# print(np.percentile(lengths_summaries.counts, 90))
# print(np.percentile(lengths_summaries.counts, 95))
# print(np.percentile(lengths_summaries.counts, 99))

def unk_counter(sentence):
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


sorted_summaries = []
sorted_texts = []
max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

sorted_summaries_text = []
sorted_texts_text = []

# beam_width = 10

for length in range(min(lengths_texts.counts), max_text_length):
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
        ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])
            sorted_summaries_text.append(clean_summaries[count])
            sorted_texts_text.append(clean_texts[count])


# print(len(sorted_summaries))
# print(len(sorted_texts))

def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


def process_encoding_input(target_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob=keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_summary_length):
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                              output_time_major=False,
                                                              impute_finished=True,
                                                              maximum_iterations=max_summary_length)
    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    # inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = dec_cell,
    #                                                        embedding = embeddings,
    #                                                       start_tokens = start_tokens,
    #                                                      end_token = end_token,
    #                                                     initial_state = initial_state,
    #                                                    beam_width = beam_width,
    #                                                   output_layer = output_layer,
    #                                                  length_penalty_weight = 0.0)

    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_summary_length)

    return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob=keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     text_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attn_mech,
                                                   attention_layer_size=rnn_size)

    initial_state = dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_state[0])

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  summary_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        # initial_state = tf.contrib.seq2seq.tile_batch(enc_state[0], multiplier=beam_width)
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)

    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                       embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       text_length,
                                                       summary_length,
                                                       max_summary_length,
                                                       rnn_size,
                                                       vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers)

    return training_logits, inference_logits


def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size):
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


# epochs = 100
epochs = 1
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int) + 1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size)

    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')
    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

start = 0
end = (len(sorted_summaries) // 10) * 7
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:", len(sorted_texts_short[-1]))

learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20
stop_early = 0
stop = 3
per_epoch = 3
update_check = (len(sorted_texts_short) // batch_size // per_epoch) - 1

update_loss = 0
batch_loss = 0
summary_update_loss = []

# C:\Users\khanh-bk\AppData\Local\Programs\Python\Python36

checkpoint = "D:/Source/PycharmProjects/best_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    # If we want to continue training a previous session
    # loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
    # loader.restore(sess, checkpoint)

    for epoch_i in range(1, epochs + 1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 summary_length: summaries_lengths,
                 text_length: texts_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(sorted_texts_short) // batch_size,
                              batch_loss / display_step,
                              batch_time * display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss / update_check, 3))
                summary_update_loss.append(update_loss)

                # If the update loss is at a new minimum, save the model
                if update_loss <= min(summary_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0

        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate

        if stop_early == stop:
            print("Stopping Training.")
            break


def text_to_seq(text):
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


# random = np.random.randint(0,len(clean_texts))

start = (len(sorted_summaries) // 10) * 7
end = len(sorted_summaries)
sorted_summaries_short = sorted_summaries_text[start:end]
sorted_texts_short = sorted_texts_text[start:end]
output_predict = []

# input_sentence = clean_texts[random]
# text = text_to_seq(clean_texts[random])

pad = vocab_to_int["<PAD>"]
count = 0
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    for input_sentence in sorted_texts_short:
        count += 1
        text = text_to_seq(input_sentence)
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          summary_length: [np.random.randint(5, 8)],
                                          text_length: [len(text)] * batch_size,
                                          keep_prob: 1.0})[0]

        output_predict.append(" ".join([int_to_vocab[i] for i in answer_logits if i != pad]))
        if (count // (len(sorted_texts_short) // 100) % 10 == 0):
            print('The data of test set was run through {}%'.format(count // (len(sorted_texts_short) // 100)))

            # pad = vocab_to_int["<PAD>"]

            # print('Original Text:', input_sentence)
            # print('\nText')
            # print('  Word Ids:    {}'.format([i for i in text]))
            # print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

            # print('\nSummary')
            # print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
            # print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _lcs(x, y):
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _len_lcs(x, y):
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _recon_lcs(x, y):
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):

        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    m = len(_split_into_words(reference_sentences))

    n = len(_split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences:
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                          ref_s)
    return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
    # hyps_and_refs = zip(hypotheses, references)
    # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
    # hypotheses, references = zip(*hyps_and_refs)

    rouge_1 = [
        rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

    rouge_2 = [
        rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

    rouge_l = [
        rouge_l_sentence_level([hyp], [ref])
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

    return rouge_1_f, rouge_1_r, rouge_1_p, rouge_2_f, rouge_2_r, rouge_2_p, rouge_l_f, rouge_l_r, rouge_l_p


rouge_1_f = 0
rouge_1_r = 0
rouge_1_p = 0
rouge_2_f = 0
rouge_2_r = 0
rouge_2_p = 0
rouge_l_f = 0
rouge_l_r = 0
rouge_l_p = 0
rouge_1_f_tmp = 0
rouge_1_r_tmp = 0
rouge_1_p_tmp = 0
rouge_2_f_tmp = 0
rouge_2_r_tmp = 0
rouge_2_p_tmp = 0
rouge_l_f_tmp = 0
rouge_l_r_tmp = 0
rouge_l_p_tmp = 0

count = 0
for summarization in sorted_summaries_short:
    rouge_1_f_tmp, rouge_1_r_tmp, rouge_1_p_tmp, rouge_2_f_tmp, rouge_2_r_tmp, rouge_2_p_tmp, rouge_l_f_tmp, rouge_l_r_tmp, rouge_l_p_tmp = rouge(
        output_predict[count], summarization)
    count += 1
    rouge_1_f += rouge_1_f_tmp
    rouge_1_r += rouge_1_r_tmp
    rouge_1_p += rouge_1_p_tmp
    rouge_2_f += rouge_2_f_tmp
    rouge_2_r += rouge_2_r_tmp
    rouge_2_p += rouge_2_p_tmp
    rouge_l_f += rouge_l_f_tmp
    rouge_l_r += rouge_l_r_tmp
    rouge_l_p += rouge_l_p_tmp

rouge_1_f = rouge_1_f / count
rouge_1_r = rouge_1_r / count
rouge_1_p = rouge_1_p / count
rouge_2_f = rouge_2_f / count
rouge_2_r = rouge_2_r / count
rouge_2_p = rouge_2_p / count
rouge_l_f = rouge_l_f / count
rouge_l_r = rouge_l_r / count
rouge_l_p = rouge_l_p / count

print('ROUGE: ')
print()
print('   ROUGE-1:')
print('      ROUGE-1/Precision:   {}'.format(rouge_1_p))
print('      ROUGE-1/Recall:      {}'.format(rouge_1_r))
print('      ROUGE-1/F1-Score:    {}'.format(rouge_1_f))
print()
print('   ROUGE-2:')
print('      ROUGE-2/Precision:   {}'.format(rouge_2_p))
print('      ROUGE-2/Recall:      {}'.format(rouge_2_r))
print('      ROUGE-2/F1-Score:    {}'.format(rouge_2_f))
print()
print('   ROUGE-L:')
print('      ROUGE-L/Precision:   {}'.format(rouge_l_p))
print('      ROUGE-L/Recall:      {}'.format(rouge_l_r))
print('      ROUGE-L/F1-Score:    {}'.format(rouge_l_f))

print('Example summarizer do: ')
for i in range(5):
    print('Example #{}'.format(i + 1))
    print('Predict: {}'.format(output_predict[i]))
    print()
    print('References: {}'.format(sorted_summaries_short[i]))
    print()
    print('Texts: {}'.format(sorted_texts_short[i]))

print()
print('The program has ended')