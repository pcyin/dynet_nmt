from __future__ import print_function

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
from bleu import calc_bleu, calc_f1

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynet-gpu', action='store_true', default=False)
    parser.add_argument('--dynet-mem', default=4000, type=int)
    parser.add_argument('--dynet-seed', default=914808182, type=int)

    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--train_mode', choices=['ml', 'rl'], default='ml')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--sample_size', default=10, type=int)
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--attention_size', default=256, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--update_every_iter', default=2, type=int)

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
    parser.add_argument('--save_model_after', default=2)
    parser.add_argument('--save_to_file', default=None, type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_niter', default=-1, type=int)

    parser.add_argument('--reward', default='bleu')

    args = parser.parse_args()
    np.random.seed(args.dynet_seed * 13 / 7)

    if args.dynet_gpu:  # the python gpu switch.
        import _gdynet as dy

    return args


def get_reward_func():
    if args.reward == 'bleu':
        return calc_bleu
    elif args.reward == 'bleu_no_bp':
        return lambda ref, hyp: calc_bleu(ref, hyp, bp=False)
    elif args.reward == 'f1':
        return calc_f1
    else:
        raise RuntimeError('unidentified reward function [%s]' % args.reward)


reward_func = None


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
    print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq), len(non_singletons)))

    top_k_words = sorted(non_singletons, reverse=True, key=word_freq.get)[:cutoff - len(vocab)]
    for word in top_k_words:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def build_id2word_vocab(vocab):
    return {v: k for k, v in vocab.iteritems()}


def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


class Hypothesis(object):
    def __init__(self, state, y, ctx_tm1, score):
        self.state = state
        self.y = y
        self.ctx_tm1 = ctx_tm1
        self.score = score


class NMT(object):
    # define dynet model for the encoder-decoder model
    def __init__(self, args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word, load_from=None, load_mode=None):
        self.args = args

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_vocab_id2word = src_vocab_id2word
        self.tgt_vocab_id2word = tgt_vocab_id2word

        model = self.model = dy.Model()

        if load_from is None:
            self.create_ml_parameters()
            self.create_rl_parameters()
        else:
            assert load_mode in ['ml', 'rl']
            params = model.load(load_from)

            if load_mode == 'ml':
                self.src_lookup, self.tgt_lookup, \
                self.enc_forward_builder, self.enc_backward_builder, self.dec_builder, \
                self.W_y, self.b_y, \
                self.W_h, self.b_h, \
                self.W_s, self.b_s, \
                self.W1_att_f, self.W1_att_e, self.W2_att = params

                self.create_rl_parameters()
            elif load_mode == 'rl':
                self.src_lookup, self.tgt_lookup, \
                self.enc_forward_builder, self.enc_backward_builder, self.dec_builder, \
                self.W_y, self.b_y, \
                self.W_h, self.b_h, \
                self.W_s, self.b_s, \
                self.W1_att_f, self.W1_att_e, self.W2_att, \
                self.W1_b, self.b1_b = params
                # self.W1_b, self.b1_b, self.W2_b, self.b2_b = params

        # set model parameters

        # set recurrent dropout
        # if args.dropout > 0.:
        #     self.enc_forward_builder.set_dropout(args.dropout)
        #     self.enc_backward_builder.set_dropout(args.dropout)
        #     self.dec_builder.set_dropout(args.dropout)

        self.ml_params = [self.src_lookup, self.tgt_lookup,
                          self.enc_forward_builder, self.enc_backward_builder, self.dec_builder,
                          self.W_y, self.b_y,
                          self.W_h, self.b_h,
                          self.W_s, self.b_s,
                          self.W1_att_f, self.W1_att_e, self.W2_att]

        # self.rl_params = [self.W1_b, self.b1_b, self.W2_b, self.b2_b]
        # single layer NN as the baseline
        self.rl_params = [self.W1_b, self.b1_b]

        print('number of parameters in the model: %d' % model.pl(), file=sys.stderr)

    def create_ml_parameters(self):
        model = self.model

        self.src_lookup = self.model.add_lookup_parameters((args.src_vocab_size, args.embed_size))
        self.tgt_lookup = self.model.add_lookup_parameters((args.tgt_vocab_size, args.embed_size))

        self.enc_forward_builder = dy.LSTMBuilder(1, args.embed_size, args.hidden_size, model)
        self.enc_backward_builder = dy.LSTMBuilder(1, args.embed_size, args.hidden_size, model)
        self.dec_builder = dy.LSTMBuilder(1, args.embed_size + args.hidden_size * 2, args.hidden_size, model)

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

    def create_rl_parameters(self):
        model = self.model

        # baseline for REINFROCE
        self.W1_b = model.add_parameters((1, args.hidden_size))
        self.b1_b = model.add_parameters((1))
        self.b1_b.zero()
        # self.W2_b = model.add_parameters((1, 50))
        # self.b2_b = model.add_parameters((1))
        # self.b2_b.zero()

    def encode(self, src_sents):
        dy.renew_cg()
        # dy.renew_cg(immediate_compute=True, check_validity=True)

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
        W1_att_f = dy.parameter(self.W1_att_f)

        completed_hypotheses = []
        decoder_init_cell = W_s * decoder_init + b_s
        hypotheses = [Hypothesis(state=self.dec_builder.initial_state([decoder_init_cell, dy.tanh(decoder_init_cell)]),
                                 y=[self.tgt_vocab['<s>']],
                                 ctx_tm1=dy.vecInput(self.args.hidden_size * 2),
                                 score=0.)]

        # pre-compute transformations for source sentences in calculating attention score
        # src_encoding_size, src_sent_len, batch_size
        src_enc_all = dy.concatenate_cols(src_encodings)
        # att_hidden_size, src_sent_len, batch_size
        src_trans_att = W1_att_f * src_enc_all
        src_len = len(src_encodings)

        t = 0
        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            t += 1
            new_hyp_scores_list = []
            for hyp in hypotheses:
                y_tm1_embed = dy.lookup(self.tgt_lookup, hyp.y[-1])
                x = dy.concatenate([y_tm1_embed, hyp.ctx_tm1])

                hyp.state = hyp.state.add_input(x)
                h_t = hyp.state.output()
                ctx_t, alpha_t = self.attention(src_enc_all, src_trans_att, h_t)

                # read_out = dy.tanh(W_h * dy.concatenate([h_t, ctx_t]) + b_h)
                read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
                # y_t = W_y * read_out + b_y
                y_t = dy.affine_transform([b_y, W_y, read_out])
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

    def sample_with_loss(self, src_encodings, decoder_init, src_batch_size, sample_num=5, to_word=False):
        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_h = dy.parameter(self.W_h)
        b_h = dy.parameter(self.b_h)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_f = dy.parameter(self.W1_att_f)

        W1_b = dy.parameter(self.W1_b)
        b1_b = dy.parameter(self.b1_b)

        # (hidden_size, batch_size)
        decoder_init_cell = W_s * decoder_init + b_s

        batch_size = src_batch_size * sample_num

        # (hidden_size, batch_size)
        decoder_init_cell_tiled = dy.concatenate_cols([decoder_init_cell for _ in xrange(sample_num)])
        decoder_init_cell = dy.reshape(decoder_init_cell_tiled, (args.hidden_size, ),
                                       batch_size=batch_size)

        decoder_init_state = dy.tanh(decoder_init_cell)

        ctx_tm1 = dy.zeroes((args.hidden_size * 2, ), batch_size=batch_size)
        s = self.dec_builder.initial_state([decoder_init_cell, decoder_init_state])

        # pre-compute transformations for source sentences in calculating attention score
        # (src_encoding_size, src_sent_len, src_batch_size)
        src_enc_all = dy.concatenate_cols(src_encodings)

        # (src_encoding_size, src_sent_len, batch_size)
        src_enc_all_tiled = dy.concatenate_cols([src_enc_all for _ in xrange(sample_num)])
        src_enc_all = dy.reshape(src_enc_all_tiled, (args.hidden_size * 2, len(src_encodings)),
                                 batch_size=batch_size)

        # (att_hidden_size, src_sent_len, batch_size)
        src_trans_att = W1_att_f * src_enc_all

        # # (att_hidden_size, src_sent_len, batch_size)
        # src_trans_att_tiled = dy.concatenate_cols([src_trans_att for _ in xrange(sample_num)])
        # src_trans_att = dy.reshape(src_trans_att_tiled, (args.attention_size, len(src_encodings)),
        #                            batch_size=batch_size)

        bos = self.tgt_vocab['<s>']
        eos = self.tgt_vocab['</s>']

        samples = [[bos for _ in xrange(batch_size)]]
        completed_samples = [list() for _ in xrange(batch_size)]
        losses = []
        baselines = []
        sample_masks = []
        num_active_words = batch_size

        t = 0
        while num_active_words > 0 and t < args.decode_max_time_step:
            t += 1

            y_tm1 = samples[-1]
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, y_tm1)
            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_enc_all, src_trans_att, h_t)

            read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
            # affine transformation tends to give (xxx, 1, batch_size) outputs
            p_t = dy.softmax(dy.affine_transform([b_y, W_y, read_out])).npvalue().reshape((-1, batch_size))

            # generate samples!
            sampled_words_t = []
            mask_t = []
            num_active_words = 0
            for sid, prev_word in enumerate(y_tm1):
                if prev_word != eos:
                # draw a sample
                    # y_t = np.random.choice(tgt_word_ids, p=p_t[:, sid] / p_t[:, sid].sum())
                    y_t = categorical_sample(p_t[:, sid])
                    sampled_words_t.append(y_t)
                    mask_t.append(1)

                    if y_t != eos:
                        num_active_words += 1
                    else:
                        # we have a completed sample
                        completed_samples[sid] = [samples[ti][sid] for ti in xrange(t)] + [y_t]
                else:
                    sampled_words_t.append(eos)
                    mask_t.append(0)

            # compute the softmax for calculating the loss function
            if args.dropout > 0.:
                read_out_train = dy.dropout(read_out, args.dropout)
            else:
                read_out_train = read_out

            y_t_weights = dy.affine_transform([b_y, W_y, read_out_train])
            # reference y_t's are sampled target words (y_t)
            loss_t = dy.pickneglogsoftmax_batch(y_t_weights, sampled_words_t)

            # compute the baseline
            b_t = self.get_rl_baseline(h_t, W1_b, b1_b)

            losses.append(loss_t)
            baselines.append(b_t)
            sample_masks.append(mask_t)

            # ending current iteration at t
            samples.append(sampled_words_t)
            ctx_tm1 = ctx_t

        if to_word:
            completed_samples = word2id(completed_samples, self.tgt_vocab_id2word)

        return completed_samples, sample_masks, losses, baselines

    def sample(self, src_sent, sample_num=5, to_word=False):
        if not type(src_sent[0]) == list:
            src_sent = [src_sent]

        src_encodings, decoder_init = self.encode(src_sent)

        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_h = dy.parameter(self.W_h)
        b_h = dy.parameter(self.b_h)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_f = dy.parameter(self.W1_att_f)

        decoder_init_cell = W_s * decoder_init + b_s
        decoder_init_state = dy.tanh(decoder_init_cell)

        # decoder_init_cell = dy.reshape(dy.concatenate_cols([decoder_init_cell for _ in xrange(sample_num)]), (args.hidden_size, ), batch_size=sample_num)
        # decoder_init_state = dy.reshape(dy.concatenate_cols([decoder_init_state for _ in xrange(sample_num)]), (args.hidden_size, ), batch_size=sample_num)

        # (hidden_size, sample_num)
        decoder_init_cell = dy.inputTensor(np.tile(decoder_init_cell.npvalue(), (sample_num, 1)).T, batched=True)
        decoder_init_state = dy.inputTensor(np.tile(decoder_init_state.npvalue(), (sample_num, 1)).T, batched=True)

        ctx_tm1 = dy.zeroes((args.hidden_size * 2, ), batch_size=sample_num)
        s = self.dec_builder.initial_state([decoder_init_cell, decoder_init_state])

        samples = [[self.tgt_vocab['<s>'] for i in xrange(sample_num)]]
        completed_samples = []
        tgt_word_ids = range(args.tgt_vocab_size)
        eos = self.tgt_vocab['</s>']

        # pre-compute transformations for source sentences in calculating attention score
        # src_encoding_size, src_sent_len, batch_size
        src_enc_all = dy.concatenate_cols(src_encodings)
        # att_hidden_size, src_sent_len, batch_size
        src_trans_att = W1_att_f * src_enc_all

        t = 0
        while len(completed_samples) < sample_num and t < args.decode_max_time_step:
            t += 1

            y_tm1 = samples[-1]
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, y_tm1)
            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_enc_all, src_trans_att, h_t)

            read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
            # affine transformation tends to give (xxx, 1, batch_size) outputs
            p_t = dy.softmax(dy.affine_transform([b_y, W_y, read_out])).npvalue().reshape((-1, sample_num))

            cur_samples = []
            for sid, prev_word in enumerate(y_tm1):
                # draw a sample
                if prev_word != eos:
                    # y_t = np.random.choice(tgt_word_ids, p=p_t[:, sid] / p_t[:, sid].sum())
                    y_t = categorical_sample(p_t[:, sid])
                    cur_samples.append(y_t)
                    if y_t == eos:
                        completed_samples.append([samples[i][sid] for i in xrange(t)] + [y_t])
                else:
                    cur_samples.append(eos)

            samples.append(cur_samples)
            ctx_tm1 = ctx_t

        if to_word:
            completed_samples = word2id(completed_samples, self.tgt_vocab_id2word)

        return completed_samples

    def get_decode_loss(self, src_encodings, decoder_init, tgt_sents):
        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_h = dy.parameter(self.W_h)
        b_h = dy.parameter(self.b_h)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_f = dy.parameter(self.W1_att_f)

        tgt_words, tgt_masks = input_transpose(tgt_sents)
        batch_size = len(tgt_sents)

        decoder_init_cell = W_s * decoder_init + b_s
        # initialize decoder state
        s = self.dec_builder.initial_state([decoder_init_cell, dy.tanh(decoder_init_cell)])
        # initialize first context vector
        ctx_tm1 = dy.vecInput(self.args.hidden_size * 2)
        # pre-compute transformations for source sentences in calculating attention score
        # src_encoding_size, src_sent_len, batch_size
        src_enc_all = dy.concatenate_cols(src_encodings)
        # att_hidden_size, src_sent_len, batch_size
        src_trans_att = W1_att_f * src_enc_all
        src_len = len(src_encodings)

        losses = []

        # start from <S>, until y_{T-1}
        for t, (y_ref_t, mask_t) in enumerate(zip(tgt_words[1:], tgt_masks[1:]), start=1):
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, tgt_words[t - 1])
            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_enc_all, src_trans_att, h_t)

            # read_out = dy.tanh(W_h * dy.concatenate([h_t, ctx_t]) + b_h)
            read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
            if args.dropout > 0.:
                read_out = dy.dropout(read_out, args.dropout)
            y_t = dy.affine_transform([b_y, W_y, read_out])
            loss_t = dy.pickneglogsoftmax_batch(y_t, y_ref_t)

            if 0 in mask_t:
                mask_expr = dy.inputVector(mask_t)
                mask_expr = dy.reshape(mask_expr, (1, ), batch_size)
                loss_t = loss_t * mask_expr

            losses.append(loss_t)
            ctx_tm1 = ctx_t

        loss = dy.esum(losses)
        # loss = dy.sum_batches(loss) / batch_size

        return loss

    def get_rl_sample_loss(self, src_encodings, decoder_init, tgt_sents, rewards):
        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_h = dy.parameter(self.W_h)
        b_h = dy.parameter(self.b_h)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_f = dy.parameter(self.W1_att_f)

        W1_b = dy.parameter(self.W1_b)
        b1_b = dy.parameter(self.b1_b)
        # W2_b = dy.parameter(self.W2_b)
        # b2_b = dy.parameter(self.b2_b)

        tgt_words, tgt_masks = input_transpose(tgt_sents)
        batch_size = len(tgt_sents)

        decoder_init_cell = W_s * decoder_init + b_s
        s = self.dec_builder.initial_state([decoder_init_cell, dy.tanh(decoder_init_cell)])
        ctx_tm1 = dy.vecInput(self.args.hidden_size * 2)
        losses = []
        losses_b = []

        # pre-compute transformations for source sentences in calculating attention score
        # src_encoding_size, src_sent_len, batch_size
        src_enc_all = dy.concatenate_cols(src_encodings)
        # att_hidden_size, src_sent_len, batch_size
        src_trans_att = W1_att_f * src_enc_all
        src_len = len(src_encodings)

        # start from <S>, until y_{T-1}
        for t, (y_ref_t, mask_t) in enumerate(zip(tgt_words[1:], tgt_masks[1:]), start=1):
            y_tm1_embed = dy.lookup_batch(self.tgt_lookup, tgt_words[t - 1])
            x = dy.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attention(src_enc_all, src_trans_att, h_t)

            # read_out = dy.tanh(W_h * dy.concatenate([h_t, ctx_t]) + b_h)
            read_out = dy.tanh(dy.affine_transform([b_h, W_h, dy.concatenate([h_t, ctx_t])]))
            if args.dropout > 0.:
                read_out = dy.dropout(read_out, args.dropout)
            y_t = dy.affine_transform([b_y, W_y, read_out])
            loss_t = dy.pickneglogsoftmax_batch(y_t, y_ref_t)

            # feed in rewards
            rewards_expr = dy.inputVector(rewards[t])
            rewards_expr = dy.reshape(rewards_expr, (1,), batch_size)

            # compute baseline
            r_b = self.get_rl_baseline(h_t, W1_b, b1_b)
            # objective for baseline - MSE
            loss_b = dy.square(r_b - rewards_expr)

            if 0 in mask_t:
                mask_expr = dy.inputVector(mask_t)
                mask_expr = dy.reshape(mask_expr, (1, ), batch_size)

                loss_t = loss_t * mask_expr
                loss_b = loss_b * mask_expr

            loss_t = (rewards_expr - dy.nobackprop(r_b)) * loss_t

            losses.append(loss_t)
            losses_b.append(loss_b)

            ctx_tm1 = ctx_t

        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / batch_size

        loss_b = dy.sum_batches(dy.esum(losses_b)) / np.sum(tgt_masks)

        return loss, loss_b

    def get_rl_baseline(self, h_t, *params):
        W1_b, b1_b = params

        h_t = dy.nobackprop(h_t)
        b = dy.tanh(W1_b * h_t + b1_b)

        return b

    def attention(self, src_enc_all, src_trans_att, h_t):
        W1_att_e = dy.parameter(self.W1_att_e)
        W2_att = dy.parameter(self.W2_att)

        att_hidden = dy.tanh(dy.colwise_add(src_trans_att, W1_att_e * h_t))

        att_weights = dy.transpose(W2_att * att_hidden) # dy.reshape(W2_att * att_hidden, (src_len, ), batch_size)
        # src_sent_len, batch_size
        att_weights = dy.softmax(att_weights)

        ctx = src_enc_all * att_weights

        return ctx, att_weights

    def get_encdec_loss(self, src_sents, tgt_sents):
        src_encodings, decoder_init = self.encode(src_sents)
        loss = self.get_decode_loss(src_encodings, decoder_init, tgt_sents)

        return loss

    def get_rl_loss(self, src_sents, tgt_sents):
        rewards = []
        loss_src_sents = []
        loss_tgt_sents = []
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            # beam_samples = self.translate(src_sent)
            tgt_samples = self.sample(src_sent, sample_num=args.sample_size, to_word=False)
            # tgt_samples_words = word2id(tgt_samples, self.tgt_vocab_id2word)

            # print '****** beam search results ******'
            # for hyp in beam_samples:
            #     print ' '.join(hyp.y)
            # print '****** sampled results ******'
            # for hyp in tgt_samples_words:
            #     print ' '.join(hyp)

            for hyp in tgt_samples:
                # reward = sentence_bleu([tgt_sent], hyp)
                reward = get_rl_reward(tgt_sent, hyp)
                loss_src_sents.append(src_sent)
                loss_tgt_sents.append(hyp)
                rewards.append(reward)

        rewards, _ = input_transpose(rewards, end_token=0)
        # compute loss
        src_encodings, decoder_init = self.encode(loss_src_sents)
        loss, loss_b = self.get_rl_sample_loss(src_encodings, decoder_init, loss_tgt_sents, rewards)

        return loss, loss_b

    def get_rl_loss_new(self, src_sents, tgt_sents):
        src_batch_size = len(src_sents)
        batch_size = src_batch_size * args.sample_size
        src_encodings, decoder_init = self.encode(src_sents)
        samples, sample_masks, losses, baselines = self.sample_with_loss(src_encodings, decoder_init,
                                                                  src_batch_size=src_batch_size,
                                                                  sample_num=args.sample_size,
                                                                  to_word=False)

        rewards = []
        for example_id, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
            offset = example_id * args.sample_size
            tgt_samples = samples[offset: offset + args.sample_size]

            for hyp in tgt_samples:
                reward = get_rl_reward(tgt_sent, hyp)
                rewards.append(reward)

        rewards, _ = input_transpose(rewards, end_token=0)

        # compute loss
        max_t = len(losses)
        masked_losses = []
        masked_losses_b = []
        for t in xrange(max_t):
            mask_t = sample_masks[t]
            loss_t = losses[t]
            b_t = baselines[t]
            reward_t = rewards[t]

            # objective for baseline - MSE
            reward_expr = dy.inputTensor(reward_t, batched=True)
            loss_bt = dy.square(b_t - reward_expr)

            # subtract loss with baseline
            loss_t = (reward_expr - b_t) * loss_t

            if 0 in mask_t:
                mask_expr = dy.inputTensor(mask_t, batched=True)

                loss_t = loss_t * mask_expr
                loss_bt = loss_bt * mask_expr

            masked_losses.append(loss_t)
            masked_losses_b.append(loss_bt)

        loss = dy.esum(masked_losses)
        loss = dy.sum_batches(loss) / batch_size

        loss_b = dy.sum_batches(dy.esum(masked_losses_b)) / np.sum(sample_masks)

        return loss, loss_b


    def save(self, path, mode='ml'):
        assert mode in ['ml', 'rl']
        if mode == 'ml':
            print('save parameters related to maximum likelihood training to %s' % path, file=sys.stderr)
            self.model.save(path, self.ml_params)
        elif mode == 'rl':
            print('save all model parameters to %s' % path, file=sys.stderr)
            self.model.save(path, self.ml_params + self.rl_params)


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


def input_transpose(sents, end_token=2):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else end_token for k in xrange(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in xrange(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def get_rl_reward(ref_sent, hyp_sent):
    reward = []
    prev_score = 0.
    delta_scores = []
    for l in xrange(1, len(hyp_sent) + 1):
        partial_hyp = hyp_sent[:l]
        y_t = hyp_sent[l - 1]
        score = reward_func(ref_sent, partial_hyp)
        # score = calc_bleu(ref_sent, partial_hyp, bp=False)
        # score = calc_f1(ref_sent, partial_hyp)

        delta_score = score - prev_score
        delta_scores.append(delta_score)
        prev_score = score

    cum_reward = 0.
    for i in reversed(xrange(len(hyp_sent))):
        reward_i = delta_scores[i]
        cum_reward += reward_i
        reward.append(cum_reward)

    reward = list(reversed(reward))

    return reward


def get_optimizer(model):
    if args.optimizer == 'adam':
        trainer = dy.AdamTrainer(model.model, alpha=args.lr)
    elif args.optimizer == 'sgd':
        trainer = dy.SimpleSGDTrainer(model.model, args.lr)
    else:
        raise RuntimeError('Unidentified optimizer %s' % args.optimizer)

    return trainer

def train_mle(args):
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
    train_iter = patience = cum_loss = cum_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = time.time()
    print('begin Maximum Likelihood training')
    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            src_sents_wids = word2id(src_sents, src_vocab)
            tgt_sents_wids = word2id(tgt_sents, tgt_vocab)
            batch_size = len(src_sents)

            if train_iter % args.valid_niter == 0:
                valid_num += 1
                print('epoch %d, iter %d, cum. loss %f, ' \
                      'cum. examples %d, time elapsed %f(s)' % (epoch, train_iter,
                                                                cum_loss / cum_examples,
                                                                cum_examples,
                                                                time.time() - train_time), file=sys.stderr)

                print('begin validation ...', file=sys.stderr)
                dev_hyps, dev_bleu = decode(model, dev_data)
                print('validation: iter %d, dev. bleu %f' % (train_iter, dev_bleu), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or dev_bleu > max(hist_valid_scores)
                hist_valid_scores.append(dev_bleu)

                if is_better:
                    patience = 0
                    best_model_iter = train_iter
                    print('save currently the best model ..', file=sys.stderr)
                    model.save(args.save_to, mode='ml')
                else:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        print('early stop!')
                        print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                        exit(0)

                if valid_num > args.save_model_after:
                    model_file = args.save_to + ('.iter%d' % train_iter)
                    print('save model to %s' % model_file, file=sys.stderr)
                    model.save(model_file, mode='ml')

                train_time = time.time()
                cum_loss = cum_examples = 0.

            loss = model.get_encdec_loss(src_sents_wids, tgt_sents_wids)
            loss = dy.sum_batches(loss) / batch_size
            loss_val = loss.value()

            cum_loss += loss_val * batch_size
            cum_examples += batch_size

            ppl = np.exp(loss_val * batch_size / sum(len(s) for s in tgt_sents))
            print('epoch %d, iter %d, loss=%f, ppl=%f' % (epoch, train_iter, loss_val, ppl))

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

    if args.model:
        model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word, args.model, load_mode='ml')
    else:
        model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)

    trainer = get_optimizer(model)

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)
    train_iter = patience = cum_loss = cum_baseline_loss = cum_examples = epoch = valid_num = update_batch = 0
    hist_valid_scores = []

    print('begin REINFORCE training')
    while args.max_niter == -1 or train_iter < args.max_niter:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            update_batch += 1
            src_sents_wids = word2id(src_sents, src_vocab)
            tgt_sents_wids = word2id(tgt_sents, tgt_vocab)
            batch_size = len(src_sents)

            if train_iter % args.valid_niter == 0:
                valid_num += 1
                print('epoch %d, iter %d, begin validation ...' % (epoch, train_iter), file=sys.stderr)

                dev_hyps, dev_bleu = decode(model, dev_data)

                print('validation: iter %d, dev. bleu %f' % (train_iter, dev_bleu), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or dev_bleu > max(hist_valid_scores)
                hist_valid_scores.append(dev_bleu)

                if is_better:
                    patience = 0
                    print('save currently the best model ..', file=sys.stderr)
                    model.save(args.save_to, mode='rl')
                else:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        print('early stop!')
                        exit(0)

            loss_rl, loss_b = model.get_rl_loss(src_sents_wids, tgt_sents_wids)
            loss = loss_rl + loss_b
            loss_val = loss.value()
            loss_rl_val = loss_rl.value()
            loss_b_val = loss_b.value()

            print('epoch %d, iter %d, batch size %d, batch loss %f, avg. RL loss %f, avg. baseline loss %f' %
                  (epoch, train_iter, batch_size, loss_val, loss_rl_val, loss_b_val),
                  file=sys.stderr)

            loss.backward()

            if update_batch % args.update_every_iter == 0:
                print('iter %d, update trainer' % train_iter)
                trainer.update()
                update_batch = 0


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
        print('*' * 50)
        print('Source: ', ' '.join(src_sent))
        print('Target: ', ' '.join(tgt_sent))
        print('Hypothesis: ', ' '.join(hyp.y))

    elapsed = time.time() - begin_time
    bleu_score = get_bleu([tgt for src, tgt in data], hypotheses)

    print('decoded %d examples, took %d s' % (len(data), elapsed), file=sys.stderr)
    if args.save_to_file:
        print('save decoding results to %s' % args.save_to_file, file=sys.stderr)
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

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word, args.model, args.train_mode)

    test_data = zip(test_data_src, test_data_tgt)

    hypotheses, bleu_score = decode(model, test_data)

    bleu_score = get_bleu([tgt for src, tgt in test_data], hypotheses)
    print('Corpus Level BLEU: %f' % bleu_score, file=sys.stderr)


if __name__ == '__main__':
    args = init_config()
    reward_func = get_reward_func()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        if args.train_mode == 'ml':
            train_mle(args)
        elif args.train_mode == 'rl':
            train_reinforce(args)
        # cProfile.run('train_reinforce(args)', sort=2)
    elif args.mode == 'test':
        test(args)
        # cProfile.run('test(args)', sort=2)


