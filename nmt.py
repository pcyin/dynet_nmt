import cProfile
from collections import defaultdict, Counter, namedtuple
from itertools import chain
import numpy as np
import random
import sys
import argparse
import dynet as dy
import cPickle as pkl
import time
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynet-gpu', action='store_true', default=False)
    parser.add_argument('--dynet-mem', default=4000, type=int)
    parser.add_argument('--dynet-seed', default=914808182, type=int)

    parser.add_argument('--mode', choices=['train', 'test'], default='train')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--attention_size', default=256, type=int)
    parser.add_argument('--dropout', default=0., type=float)

    parser.add_argument('--src_vocab_size', default=20000, type=int)
    parser.add_argument('--tgt_vocab_size', default=20000, type=int)

    parser.add_argument('--train_src')
    parser.add_argument('--train_tgt')
    parser.add_argument('--dev_src')
    parser.add_argument('--dev_tgt')
    parser.add_argument('--test_src')
    parser.add_argument('--test_tgt')

    parser.add_argument('--decode_max_time_step', default=200, type=int)

    parser.add_argument('--valid_niter', default=500, type=int)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--save_to', default='model', type=str)
    parser.add_argument('--save_to_file', default=None, type=str)
    parser.add_argument('--patience', default=5, type=int)

    args = parser.parse_args()
    np.random.seed(args.dynet_seed * 13 / 7)

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
    non_singletons = [w for w in word_freq if word_freq[w] > 1 and w not in vocab]  # do not count <unk> in corpus
    print 'number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq), len(non_singletons))

    top_k_words = sorted(non_singletons, reverse=True, key=word_freq.get)[:cutoff - len(vocab)]
    for word in top_k_words:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def build_id2word_vocab(vocab):
    return {v: k for k, v in vocab.iteritems()}


class Hypothesis(object):
    def __init__(self, state, y, ctx_tm1, score):
        self.state = state
        self.y = y
        self.ctx_tm1 = ctx_tm1
        self.score = score

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

        self.enc_forward_builder = dy.LSTMBuilder(1, args.embed_size, args.hidden_size, model)
        self.enc_backward_builder = dy.LSTMBuilder(1, args.embed_size, args.hidden_size, model)
        self.dec_builder = dy.LSTMBuilder(1, args.embed_size + args.hidden_size * 2, args.hidden_size, model)

        # set recurrent dropout
        if args.dropout > 0.:
            self.enc_forward_builder.set_dropout(args.dropout)
            self.enc_backward_builder.set_dropout(args.dropout)
            self.dec_builder.set_dropout(args.dropout)

        # target word embedding
        self.W_y = model.add_parameters((args.tgt_vocab_size, args.embed_size))
        self.b_y = model.add_parameters((args.tgt_vocab_size))
        self.b_y.zero()

        # transformation of decoder hidden states and context vectors before reading out target words
        self.W_h = model.add_parameters((args.embed_size, args.hidden_size + args.hidden_size * 2))
        self.b_h = model.add_parameters((args.embed_size))
        self.b_h.zero()

        # transformation of context vectors at t_0 in decoding
        self.W_s = model.add_parameters((args.hidden_size, args.hidden_size * 2))
        self.b_s = model.add_parameters((args.hidden_size))
        self.b_s.zero()

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

        forward_states = forward_state.add_inputs(src_words_embeds)
        backward_states = backward_state.add_inputs(src_words_embeds_reversed)[::-1]

        src_encodings = []
        forward_cells = []
        backward_cells = []
        for forward_state, backward_state in zip(forward_states, backward_states):
            fwd_cell, fwd_enc = forward_state.s()
            bak_cell, bak_enc = backward_state.s()

            src_encodings.append(dy.concatenate([fwd_enc, bak_enc]))
            forward_cells.append(fwd_cell)
            backward_cells.append(bak_cell)

        decoder_init = dy.concatenate([forward_cells[-1], backward_cells[0]])
        return src_encodings, decoder_init

    def translate(self, src_sent, beam_size=None, to_word=True):
        if not type(src_sent[0]) == list:
            src_sent = [src_sent]
        if not beam_size:
            beam_size = args.beam_size

        src_encodings, decoder_init = self.encode(src_sent)

        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_h = dy.parameter(self.W_h)
        b_h = dy.parameter(self.b_h)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        completed_hypotheses = []
        decoder_init_cell = W_s * decoder_init + b_s
        hypotheses = [Hypothesis(state=self.dec_builder.initial_state([decoder_init_cell, dy.tanh(decoder_init_cell)]),
                                 y=[self.tgt_vocab['<s>']],
                                 ctx_tm1=dy.vecInput(self.args.hidden_size * 2),
                                 score=0.)]

        t = 0
        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            t += 1
            new_hyp_scores_list = []
            for hyp in hypotheses:
                y_tm1_embed = dy.lookup(self.tgt_lookup, hyp.y[-1])
                x = dy.concatenate([y_tm1_embed, hyp.ctx_tm1])

                hyp.state = hyp.state.add_input(x)
                h_t = hyp.state.output()
                ctx_t, alpha_t = self.attention(src_encodings, h_t, batch_size=1)

                # read_out = dy.tanh(W_h * dy.concatenate([h_t, ctx_t]) + b_h)
                read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
                y_t = W_y * read_out + b_y
                p_t = dy.log_softmax(y_t).npvalue()

                hyp.ctx_tm1 = ctx_t

                # add the score of the current hypothesis to p_t
                new_hyp_scores = hyp.score + p_t
                new_hyp_scores_list.append(new_hyp_scores)

            live_nyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = np.concatenate(new_hyp_scores_list).flatten()
            new_hyp_pos = (-new_hyp_scores).argsort()[:live_nyp_num]
            prev_hyp_ids = new_hyp_pos / args.tgt_vocab_size
            word_ids = new_hyp_pos % args.tgt_vocab_size
            new_hyp_scores = new_hyp_scores[new_hyp_pos]

            new_hypotheses = []

            for prev_hyp_id, word_id, hyp_score in zip(prev_hyp_ids, word_ids, new_hyp_scores):
                prev_hyp = hypotheses[prev_hyp_id]
                hyp = Hypothesis(state=prev_hyp.state,
                                 y=prev_hyp.y + [word_id],
                                 ctx_tm1=prev_hyp.ctx_tm1,
                                 score=hyp_score)

                if word_id == self.tgt_vocab['</s>']:
                    completed_hypotheses.append(hyp)
                else:
                    new_hypotheses.append(hyp)

            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]

        if to_word:
            for hyp in completed_hypotheses:
                hyp.y = [self.tgt_vocab_id2word[i] for i in hyp.y]

        return sorted(completed_hypotheses, key=lambda x: x.score, reverse=True)

    def get_decode_loss(self, src_encodings, decoder_init, tgt_sents):
        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_h = dy.parameter(self.W_h)
        b_h = dy.parameter(self.b_h)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        tgt_words, tgt_masks = input_transpose(tgt_sents)
        batch_size = len(tgt_sents)

        decoder_init_cell = W_s * decoder_init + b_s
        s = self.dec_builder.initial_state([decoder_init_cell, dy.tanh(decoder_init_cell)])
        ctx_tm1 = dy.vecInput(self.args.hidden_size * 2)
        losses = []

        # start from <S>, until y_{T-1}
        for t, (y_ref_t, mask_t) in enumerate(zip(tgt_words[1:], tgt_masks[1:]), start=1):
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, tgt_words[t - 1])
            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_encodings, h_t, batch_size)

            # read_out = dy.tanh(W_h * dy.concatenate([h_t, ctx_t]) + b_h)
            read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
            if args.dropout > 0.:
                read_out = dy.dropout(read_out, args.dropout)
            y_t = W_y * read_out + b_y
            loss_t = -dy.pick_batch(dy.log(dy.softmax(y_t)), y_ref_t) # dy.pickneglogsoftmax_batch(y_t, y_ref_t)

            if 0 in mask_t:
                mask_expr = dy.inputVector(mask_t)
                mask_expr = dy.reshape(mask_expr, (1, ), batch_size)
                loss_t = loss_t * mask_expr

            losses.append(loss_t)
            ctx_tm1 = ctx_t

        loss = dy.esum(losses)
        # loss = dy.sum_batches(loss) / batch_size

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
        src_encodings, decoder_init = self.encode(src_sents)
        loss = self.get_decode_loss(src_encodings, decoder_init, tgt_sents)

        return loss

    def get_rl_loss(self, src_sents, tgt_sents):
        # sample using beam search
        rewards = []
        loss_src_sents = []
        loss_tgt_sents = []
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            tgt_samples = self.translate(src_sent, to_word=False)
            for hyp in tgt_samples:
                tgt_sent_pred = hyp.y
                reward = sentence_bleu([tgt_sent], tgt_sent_pred)

                loss_src_sents.append(src_sent)
                loss_tgt_sents.append(tgt_sent_pred)
                rewards.append(reward)

        # compute loss
        batch_size = len(rewards)
        loss = self.get_encdec_loss(loss_src_sents, loss_tgt_sents)
        r = dy.inputVector(rewards)
        r = dy.reshape(r, (1, ), batch_size)
        loss = r * loss

        loss = dy.sum_batches(loss) / batch_size

        return loss

    def load(self, path):
        print >>sys.stderr, 'loading model from: %s' % path
        self.model.load(path)


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
        np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

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
    train_iter = patience = cum_loss = cum_examples = epoch = 0
    hist_valid_scores = []
    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            src_sents_wids = word2id(src_sents, src_vocab)
            tgt_sents_wids = word2id(tgt_sents, tgt_vocab)
            batch_size = len(src_sents)

            if train_iter % args.valid_niter == 0:
                print >>sys.stderr, 'epoch %d, iter %d, cum. loss %f, cum. examples %d' % (epoch, train_iter,
                                                                                           cum_loss / cum_examples,
                                                                                           cum_examples)
                cum_loss = cum_examples = 0.
                print >>sys.stderr, 'begin validation ...'
                dev_hyps, dev_bleu = decode(model, dev_data)
                print >>sys.stderr, 'validation: iter %d, dev. bleu %f' % (train_iter, dev_bleu)

                is_better = len(hist_valid_scores) == 0 or dev_bleu > max(hist_valid_scores)
                hist_valid_scores.append(dev_bleu)

                if is_better:
                    patience = 0
                    print >>sys.stderr, 'save currently the best model ..'
                    model.model.save(args.save_to + '.bin')
                else:
                    patience += 1
                    print >>sys.stderr, 'hit patience %d' % patience
                    if patience == args.patience:
                        print 'early stop!'
                        exit(0)

            loss = model.get_encdec_loss(src_sents_wids, tgt_sents_wids)
            loss = dy.sum_batches(loss)
            loss_val = loss.value()

            cum_loss += loss_val * batch_size
            cum_examples += batch_size

            ppl = np.exp(loss_val * batch_size / sum(len(s) for s in tgt_sents))
            print 'epoch %d, iter %d, loss=%f, ppl=%f' % (epoch, train_iter, loss_val, ppl)

            loss.backward()
            trainer.update()


def train_reinforce(args):
    train_data_src = read_corpus(args.train_src)
    train_data_tgt = read_corpus(args.train_tgt)

    dev_data_src = read_corpus(args.dev_src)
    dev_data_tgt = read_corpus(args.dev_tgt)

    src_vocab = build_vocab(train_data_src, args.src_vocab_size)
    tgt_vocab = build_vocab(train_data_tgt, args.tgt_vocab_size)

    src_vocab_id2word = build_id2word_vocab(src_vocab)
    tgt_vocab_id2word = build_id2word_vocab(tgt_vocab)

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)
    if args.model:
        model.load(args.model)

    trainer = dy.AdamTrainer(model.model)

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)
    train_iter = patience = cum_loss = cum_examples = epoch = 0
    hist_valid_scores = []
    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            src_sents_wids = word2id(src_sents, src_vocab)
            tgt_sents_wids = word2id(tgt_sents, tgt_vocab)
            batch_size = len(src_sents)

            if train_iter % args.valid_niter == 0:
                print >>sys.stderr, 'epoch %d, iter %d, cum. loss %f, cum. examples %d' % (epoch, train_iter,
                                                                                           cum_loss / cum_examples,
                                                                                           cum_examples)
                cum_loss = cum_examples = 0.
                print >>sys.stderr, 'begin validation ...'
                dev_hyps, dev_bleu = decode(model, dev_data)
                print >>sys.stderr, 'validation: iter %d, dev. bleu %f' % (train_iter, dev_bleu)

                is_better = len(hist_valid_scores) == 0 or dev_bleu > max(hist_valid_scores)
                hist_valid_scores.append(dev_bleu)

                if is_better:
                    patience = 0
                    print >>sys.stderr, 'save currently the best model ..'
                    model.model.save(args.save_to + '.bin')
                else:
                    patience += 1
                    print >>sys.stderr, 'hit patience %d' % patience
                    if patience == args.patience:
                        print 'early stop!'
                        exit(0)

            loss = model.get_rl_loss(src_sents_wids, tgt_sents_wids)
            loss_val = loss.value()

            cum_loss += loss_val
            cum_examples += batch_size

            print 'epoch %d, iter %d, loss=%f' % (epoch, train_iter, loss_val)

            loss.backward()
            trainer.update()


def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
                             [hyp[1:-1] for hyp in hypotheses])

    return bleu_score

def decode(model, data):
    hypotheses = []
    begin_time = time.time()
    for src_sent, tgt_sent in data:
        src_sent_wids = word2id(src_sent, model.src_vocab)
        hyp = model.translate(src_sent_wids)[0]
        hypotheses.append(hyp.y)
        print '*' * 50
        print 'Source: ', ' '.join(src_sent)
        print 'Target: ', ' '.join(tgt_sent)
        print 'Hypothesis: ', ' '.join(hyp.y)

    elapsed = time.time() - begin_time
    bleu_score = get_bleu([tgt for src, tgt in data], hypotheses)

    print >>sys.stderr, 'decoded %d examples, took %d s' % (len(data), elapsed)
    if args.save_to_file:
        print >> sys.stderr, 'save decoding results to %s' % args.save_to_file
        with open(args.save_to_file, 'w') as f:
            for hyp in hypotheses:
                f.write(' '.join(hyp[1:-1]) + '\n')

    return hypotheses, bleu_score

def test(args):
    train_data_src = read_corpus(args.train_src)
    train_data_tgt = read_corpus(args.train_tgt)

    src_vocab = build_vocab(train_data_src, args.src_vocab_size)
    tgt_vocab = build_vocab(train_data_tgt, args.tgt_vocab_size)

    src_vocab_id2word = build_id2word_vocab(src_vocab)
    tgt_vocab_id2word = build_id2word_vocab(tgt_vocab)

    test_data_src = read_corpus(args.test_src)
    test_data_tgt = read_corpus(args.test_tgt)

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)
    model.load(args.model)

    test_data = zip(test_data_src, test_data_tgt)

    hypotheses, bleu_score = decode(model, test_data)

    bleu_score = get_bleu([tgt for src, tgt in test_data], hypotheses)
    print 'Corpus Level BLEU: %f' % bleu_score

if __name__ == '__main__':
    args = init_config()
    print >>sys.stderr, args
    if args.mode == 'train':
        # train(args)
        train_reinforce(args)
    elif args.mode == 'test':
        test(args)
        # cProfile.run('test(args)', sort=2)


