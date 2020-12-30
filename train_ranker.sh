
python reimplement_mat2gen.py --vocab_src ./data/m2g/dd/vocab.txt \
                --vocab_tgt ./data/m2g/dd/vocab.txt \
                --embed_dim 512 \
                --ff_embed_dim 1024 \
                --num_heads 8 \
                --num_layers 2 \
                --dropout 0.1 \
                --epochs 20 \
                --lr 0.0001 \
                --train_batch_size 128 \
                --dev_batch_size 128 \
                --print_every 100 \
                --eval_every 1000 \
                --train_data ./data/m2g/dd/train.txt \
                --dev_data ./data/m2g/dd/valid.txt \