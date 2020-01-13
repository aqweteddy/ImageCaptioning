# Image Captioning

## Environment

```bash
git clone https://github.com/aqweteddy/ImageCaptioning.git
cd ImageCaptioning

# install requirement

# create virtual environment
# ...

pip install -r requirements.txt
```
### Download Model

[Gdrive](https://drive.google.com/open?id=1CDAiIrU69oucRXcgRaxmyYguuFGUi6E8)

## Dataset

* `sh download.sh` to download MS COCO
<!-- [download flickr8k](https://drive.google.com/drive/folders/19jGGC1HsJRTpGIzBqA7OyO3Z-SxSFQQS?usp=sharing) -->

## How to run ?

### Train

#### Preprocess

* build dictionary
* resize images

```bash
python preprocess.py\
        --caption_path data/annotations/captions_train2014.json\
        --dict_path data/vocab.txt\
        --img_input_dir data/train2014 --img_output_dir data/train2014_resize
```

#### train

```bash
python train.py\
        --vocab_path data/vocab.txt\
        --image_dir data/train2014_resize\
        --caption_path data/annotations/captions_train2014.json\
        --save_ckpt 1000\
        --embed_size 256\
        --hidden_size 512\
        --num_layers 1\
        --num_epochs 5\
        --batch_size 64\
        --learning_rate 0.001\
	    --model_path model/
```

#### Infer one image

```py
from infer import load_model, eval1
from preprocess_text import Dictionary


vocab = Dictionary()
vocab.load_dict('path to Dictionary')
encoder, decoder = load_model(encoder_path='path_to_encder',
               decoder_path='path_to_decoder',
               vocab_size=len(vocab),
               layer_type='gru',
               embed_size=256,
               hidden_size=512,
               num_layers=2,
            )
result = eval1('path_to_image', vocab, encoder, decoder)
print(' '.join(result))
```

## Citation

* [MS COCO Dataset](https://github.com/cocodataset/cocoapi)
* [Image Caption Implement](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning)
* [nlg eval](https://github.com/Maluuba/nlg-eval)