from collections import defaultdict
import dynet as dy
import numpy as np
import random
import sys

class Attention:
    def __init__(self, model, training_src, training_tgt, ...):
        self.model = model
        self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]
        self.src_token_to_id, self.src_id_to_token = XXXX
        self.tgt_token_to_id, self.tgt_id_to_token = XXXX 

        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.l2r_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
        self.r2l_builder = builder(self.layers, self.embed_size, self.hidden_size, model)

        self.dec_builder = builder(self.layers, XXXX, self.hidden_size, model)

        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_y = model.add_parameters((self.tgt_vocab_size))

        self.W1_att_f = model.add_parameters((self.attention_size, XXXX))
        self.W1_att_e = model.add_parameters((self.attention_size, XXXX))
        self.w2_att = model.add_parameters((self.attention_size))

    # Calculates the context vector using a MLP
    # h_fs: matrix of embeddings for the source words
    # h_e: hidden state of the decoder
    def __attention_mlp(self, h_fs_matrix, h_e):
        W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        w2_att = dy.parameter(self.w2_att)

        # Calculate the alignment score vector
        # Hint: Can we make this more efficient?
        a_t = XXXX(W1_att_f, W1_att_e, w2_att, h_fs_matrix, h_e) 
        alignment = dy.softmax(a_t)
        c_t = h_fs_matrix * alignment
        return c_t

    # Training step over a single sentence pair
    def __step(self, instance):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att = dy.parameter(self.W1_att)
        w2_att = dy.parameter(self.w2_att)

        src_sent, tgt_sent = instance
        src_sent_rev = list(reversed(src_sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            l2r_state = XXXX
            r2l_state = XXXX
            l2r_contexts.append(l2r_state.output()) #[<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output()) #[</S> x_n, x_{n-1}, ... <S>]

        r2l_contexts.reverse() #[<S>, x_1, x_2, ..., </S>]

        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(XXXX(l2r_i, r2l_i))
        h_fs_matrix = XXXX(h_fs)

        losses = []
        num_words = 0

        # Decoder
        c_t = dy.vecInput(XXXX)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id['<S>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        for (cw, nw) in zip(tgt_sent, tgt_sent[1:]):
            h_e = dec_state.output()
            c_t = self.__attention_mlp(h_fs_matrix, h_e)
            # Get the embedding for the current target word
            embed_t = XXXX
            # Create input vector to the decoder
            x_t = XXXX(embed_t, c_t)
            dec_state = XXXX
            y_star = XXXX([b_y, W_y, dec_state.output()])
            loss = XXXX(y_star, XXXX)
            losses.append(loss)
            num_words += 1
 
        return dy.esum(losses), num_words

    def translate_sentence(self, sent):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att = dy.parameter(self.W1_att)
        w2_att = dy.parameter(self.w2_att)

        sent_rev = list(reversed(sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(sent, sent_rev):
            l2r_state = XXXX 
            r2l_state = XXXX
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(XXXX(l2r_i, r2l_i))

        # Decoder
        trans_sentence = ['<S>']
        cw = trans_sentence[-1]
        c_t = dy.vecInput(self.hidden_size * 2)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id['<S>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        while len(trans_sentence) < self.max_len:
            h_e = dec_state.output()
            c_t = self.__attention_mlp(h_fs, h_e)
            embed_t = XXXX
            x_t = XXXX(embed_t, c_t)
            dec_state = XXXX
            y_star = XXXX([b_y, W_y, dec_state.output()])
            p = dy.softmax(y_star)
            cw = self.tgt_id_to_token[XXXX]
            if cw == '</S>':
                break
            trans_sentence.append(cw)

        return ' '.join(trans_sentence[1:])

def main():
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    training_src = read_file(sys.argv[1])
    training_tgt = read_file(sys.argv[2])
    dev_src = read_file(sys.argv[3])
    dev_tgt = read_file(sys.argv[4])
    test_src = read_file(sys.argv[5])
    attention = Attention(model, training_src, training_tgt)

if __name__ == '__main__': main()
