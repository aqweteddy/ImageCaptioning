import json
from argparse import ArgumentParser
from nlgeval import NLGEval
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--answer_file', type=str, default='captions_val2014.json' , help='answer json path')
parser.add_argument('--model_file', type=str, default='val.txt', help='model output')
args = parser.parse_args()

print(f'Model Ouput: {args.model_file}')
print(f'Reference Ansewer: {args.answer_file}')

with open(args.answer_file, 'r') as f:
    cap_val = json.load(f)
    # cap_img = cap_val['images']
    cap_txt = cap_val['annotations']
    del cap_val
cap_txt = dict({str(cap['image_id']): cap['caption'] for cap in cap_txt})

reference = []
hypothesis = []
with open(args.model_file, 'r') as f:
    for line in tqdm(f.readlines()):
        file_name = line.split(',')[0]
        description = ''.join(line.split(',')[1:])
        for image_id, caption in cap_txt.items():
            if image_id in file_name:
                reference.append([caption.strip()])
                hypothesis.append(description.strip())
                break

print('Loading evaluate model...')
nlgeval = NLGEval()
metrics_dict = nlgeval.compute_metrics(references, hypothesis)
print(metrics_dict)