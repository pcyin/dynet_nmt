from collections import defaultdict, Counter
from itertools import chain
import numpy as np
import random
import sys
import argparse
import dynet as dy

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynet-gpu', action='store_true', default=False)
    parser.add_argument('--dynet-mem', default=4000, type=int)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embed_size', default=512, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--attention_size', default=256, type=int)

    parser.add_argument('--src_vocab_size', default=20000, type=int)
    parser.add_argument('--tgt_vocab_size', default=20000, type=int)

    parser.add_argument('--train_src')
    parser.add_argument('--train_tgt')
    parser.add_argument('--dev_src')
    parser.add_argument('--dev_tgt')

    parser.add_argument('--valid_niter', default=500, type=int)

    args = parser.parse_args()

    if args.dynet_gpu:  # the python gpu switch.
        print 'using GPU'
        import _gdynet as dy

    return args

def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def build_vocab(data, cutoff):
    vocab = defaultdict(lambda: 0)
    vocab['<unk>'] = 0
    vocab['<s>'] = 1
    vocab['</s>'] = 2

    word_freq = Counter(chain(*data))
    top_k_words = sorted(word_freq, reverse=True, key=word_freq.get)[:cutoff - len(vocab)]
    for word in top_k_words:
        freq = word_freq
        if freq >= 1 and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def build_id2word_vocab(vocab):
    return {v: k for k, v in vocab.iteritems()}


class NMT(object):
    # define dynet model for the encoder-decoder model
    def __init__(self, args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word):
        model = self.model = dy.Model()
        self.args = args
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_id2word = src_vocab_id2word
        self.tgt_vocab_id2word = tgt_vocab_id2word

        self.src_lookup = self.model.add_lookup_parameters((args.src_vocab_size, args.embed_size))
        self.tgt_lookup = self.model.add_lookup_parameters((args.tgt_vocab_size, args.embed_size))

        self.enc_forward_builder = dy.GRUBuilder(1, args.embed_size, args.hidden_size, model)
        self.enc_backward_builder = dy.GRUBuilder(1, args.embed_size, args.hidden_size, model)

        self.dec_builder = dy.GRUBuilder(1, args.embed_size + args.hidden_size * 2, args.hidden_size, model)

        self.W_y = model.add_parameters((args.tgt_vocab_size, args.hidden_size + args.hidden_size * 2))
        self.b_y = model.add_parameters((args.tgt_vocab_size))

        self.W_s = model.add_parameters((args.hidden_size, args.hidden_size * 2))
        self.b_s = model.add_parameters((args.hidden_size))

        self.W1_att_f = model.add_parameters((args.attention_size, args.hidden_size * 2))
        self.W1_att_e = model.add_parameters((args.attention_size, args.hidden_size))
        self.W2_att = model.add_parameters((1, args.attention_size))

    def encode(self, src_sents):
        dy.renew_cg()

        # bidirectional representations
        forward_state = self.enc_forward_builder.initial_state()
        backward_state = self.enc_backward_builder.initial_state()

        src_words, src_masks = input_transpose(src_sents)
        src_words_embeds = [dy.lookup_batch(self.src_lookup, wids) for wids in src_words]
        src_words_embeds_reversed = src_words_embeds[::-1]

        forward_encodings = forward_state.transduce(src_words_embeds)
        backward_encodings = backward_state.transduce(src_words_embeds_reversed)[::-1]

        src_encodings = [dy.concatenate(list(t)) for t in zip(forward_encodings, backward_encodings)]

        return src_encodings

    def translate(self, src_sent):
        if not type(src_sent[0]) == list:
            src_sent = [src_sent]

        src_encodings = self.encode(src_sent)

        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        s = self.dec_builder.initial_state([dy.tanh(W_s * src_encodings[-1] + b_s)])
        ctx_tm1 = dy.vecInput(self.args.hidden_size * 2)

        hypothesis = ['<s>']

        for t in xrange(100):
            y_tm1 = hypothesis[-1]
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, [self.tgt_vocab[y_tm1]])

            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_encodings, h_t, batch_size=1)

            y_t = dy.affine_transform([b_y, W_y, dy.concatenate([h_t, ctx_t])])
            p_t = dy.log_softmax(y_t).npvalue()

            y_t_id = np.argmax(p_t)
            y_t = self.tgt_vocab_id2word[y_t_id]

            hypothesis.append(y_t)
            ctx_tm1 = ctx_t

            if y_t == '</s>':
                break

        return hypothesis

    def get_decode_loss(self, src_encodings, tgt_sents):
        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        tgt_words, tgt_masks = input_transpose(tgt_sents)
        batch_size = len(tgt_sents)

        s = self.dec_builder.initial_state([dy.tanh(W_s * src_encodings[-1] + b_s)])
        ctx_tm1 = dy.vecInput(self.args.hidden_size * 2)
        losses = []

        # start from <S>, until y_{T-1}
        for t, (y_ref_t, mask_t) in enumerate(zip(tgt_words[1:], tgt_masks[1:]), start=1):
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, tgt_words[t - 1])
            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_encodings, h_t, batch_size)

            y_t = dy.affine_transform([b_y, W_y, dy.concatenate([h_t, ctx_t])])
            loss_t = dy.pickneglogsoftmax_batch(y_t, y_ref_t)

            if 0 in mask_t:
                mask_expr = dy.inputVector(mask_t)
                mask_expr = dy.reshape(mask_expr, (1, ), batch_size)
                loss_t = loss_t * mask_expr

            losses.append(loss_t)
            ctx_tm1 = ctx_t

        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / batch_size

        return loss

    def attention(self, src_encodings, h_t, batch_size):
        W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        W2_att = dy.parameter(self.W2_att)

        src_len = len(src_encodings)

        # enc_size, sent_len, batch_size
        src_enc_all = dy.concatenate_cols(src_encodings)

        att_hidden = dy.tanh(dy.colwise_add(W1_att_f * src_enc_all, W1_att_e * h_t))
        att_weights = dy.reshape(W2_att * att_hidden, (src_len, ), batch_size)
        # sent_len, batch_size
        att_weights = dy.softmax(att_weights)

        ctx = src_enc_all * att_weights

        return ctx, att_weights

    def get_encdec_loss(self, src_sents, tgt_sents):
        src_encodings = self.encode(src_sents)
        loss = self.get_decode_loss(src_encodings, tgt_sents)

        return loss


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        yield [data[i * batch_size + b][0] for b in range(cur_batch_size)], \
              [data[i * batch_size + b][1] for b in range(cur_batch_size)]


def data_iter(data, batch_size):
    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    while True:
        print 'new epoch'

        np.random.shuffle(batched_data)
        for batch in batched_data:
            yield batch


def input_transpose(sents, pad=True):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    # assume the id of </s> is 2
    sents_t = []
    masks = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else 2 for k in xrange(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in xrange(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def train(args):
    train_data_src = read_corpus(args.train_src)
    train_data_tgt = read_corpus(args.train_tgt)

    dev_data_src = read_corpus(args.dev_src)
    dev_data_tgt = read_corpus(args.dev_tgt)

    src_vocab = build_vocab(train_data_src, args.src_vocab_size)
    tgt_vocab = build_vocab(train_data_tgt, args.tgt_vocab_size)

    src_vocab_id2word = build_id2word_vocab(src_vocab)
    tgt_vocab_id2word = build_id2word_vocab(tgt_vocab)

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)
    trainer = dy.AdamTrainer(model.model)

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)
    train_iter = 0
    for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
        train_iter += 1
        src_sents_wids = word2id(src_sents, src_vocab)
        tgt_sents_wids = word2id(tgt_sents, tgt_vocab)
        batch_size = len(src_sents)

        if train_iter % args.valid_niter == 0:
            for i in xrange(min(10, len(dev_data))):
                dev_src_sent, dev_tgt_sent = dev_data[i]
                print 'source:', dev_src_sent
                print 'target:', dev_tgt_sent
                print 'greedy decoding:', model.translate(word2id(dev_src_sent, src_vocab))
                print '*' * 50

            model.model.save('model.bin')

        loss = model.get_encdec_loss(src_sents_wids, tgt_sents_wids)
        loss_val = loss.value()
        ppl = np.exp(loss_val * batch_size / sum(len(s) for s in tgt_sents))
        print >>sys.stderr, 'loss=%f, ppl=%f' % (loss_val, ppl)

        loss.backward()
        trainer.update()

if __name__ == '__main__':
    args = init_config()
    train(args)

