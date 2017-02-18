model_name="model.de-en"
python nmt.py \
	--dynet-mem 10000 \
    --dynet-gpu \
    --dynet-seed 914808182 \
    --mode train \
    --save_to ${model_name} \
    --valid_niter 2500 \
    --beam_size 5 \
    --train_src en-de/train.en-de.low.filt.de \
    --train_tgt en-de/train.en-de.low.filt.en \
    --dev_src en-de/valid.en-de.low.de \
    --dev_tgt en-de/valid.en-de.low.en \
    --test_src en-de/test.en-de.low.de \
    --test_tgt en-de/test.en-de.low.en 2>train.${model_name}.log
