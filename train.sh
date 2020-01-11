python train.py\
        --vocab_path data/vocab.txt\
        --image_dir data/train2014_resize\
        --caption_path data/annotations/captions_train2014.json\
        --save_ckpt 1000\
        --embed_size 256\
        --hidden_size 512\
        --num_layers 2\
        --num_epochs 5\
        --batch_size 64\
        --learning_rate 0.001\
	--model_path model/

