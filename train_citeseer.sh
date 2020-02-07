python mixhop_trainer.py --dataset_name=ind.citeseer --adj_pows=0:20,1:20,2:20 \
  --learn_rate=0.25 --lr_decrement_every=40 --early_stop_steps=200 \
  --input_dropout=0.0 --layer_dropout=0.9 --l2reg=5e-2 \
  --retrain --random_walk_approx --num_walks 500
