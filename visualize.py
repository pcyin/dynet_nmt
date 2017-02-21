import h5py
from matplotlib import pyplot as plt
import numpy as np
from nmt import read_corpus

def visualize(idx, src_sent, tgt_sent, alpha, filename):
    plt.figure(idx)    
    alpha_m = np.stack(alpha, axis=1)
    plt.imshow(alpha_m, cmap='gray', interpolation='nearest')
    print 'type(src_sent) = ', type(src_sent), type(src_sent[0])
    print 'Process {} sents'.format(idx)
    print 'Src = ', ' '.join(src_sent)
    print 'Tgt = ', ' '.join(tgt_sent)
    src_sent = [ unicode(s, errors='ignore') for s in src_sent]
    tgt_sent = [ unicode(s, errors='ignore') for s in tgt_sent]
    plt.xticks(range(0, len(tgt_sent)-1), tgt_sent[1:], rotation='vertical')
    plt.yticks(range(0, len(src_sent)-1), src_sent[1:])
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

filename = 'model.de-en.de3w.en2w.h_to_embed_space.affine_trans.dropout0.5.bin_alpha.npz'
d = np.load(open(filename, 'r'))
alphas = d['alpha']

src_sents = read_corpus('en-de/test.en-de.low.de')
tgt_sents = read_corpus('model.de-en.de3w.en2w.h_to_embed_space.affine_trans.dropout0.5.decode')

assert len(src_sents) == len(tgt_sents), 'src={}, tgt={}'.format(len(src_sents), len(tgt_sents))
assert len(alphas) == len(src_sents), 'a={}, src={}'.format(len(alphas), len(src_sents))

for idx, a in enumerate(alphas):
    visualize(idx, src_sents[idx], tgt_sents[idx], a, 'alignment/' + str(idx) + '.png')



