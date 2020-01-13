import json
from argparse import ArgumentParser
from nlgeval import NLGEval
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--answer_file', type=str, default='captions_val2014.json' , help='answer json path')
parser.add_argument('--model_file', type=str, default='val.txt', help='model output')
parser.add_argument('--hypothesis', type=str, default='hypothesis.txt', help='output hypothesis(model)')
parser.add_argument('--reference', type=str, default='reference.txt', help='output(reference)')

args = parser.parse_args()

print(f'Model Ouput: {args.model_file}')
print(f'Reference Ansewer: {args.answer_file}')

with open(args.answer_file, 'r') as f:
    cap_val = json.load(f)
    cap_txt = cap_val['annotations']
    del cap_val
cap_txt = dict({str(cap['image_id']): cap['caption'] for cap in cap_txt})

reference = []
hypothesis = []

# nlgeval = NLGEval()

with open(args.model_file, 'r') as f:
    for line in tqdm(f.readlines()):
        file_name = line.split(',')[0]
        description = ''.join(line.split(',')[1:])
        for image_id, caption in cap_txt.items():
            if image_id in file_name:
                reference.append(caption.strip())
                hypothesis.append(description.strip())
                break

with open(args.hypothesis, 'w') as f:
    for hy in hypothesis:
        f.writelines([hy, '\n'])
with open(args.reference, 'w') as f:
    for ref in reference:
        f.writelines([ref, '\n'])
# print('Loading evaluate model...')
# nlgeval = NLGEval()
# print('Evaluating...')
# assert len(reference) == len(hypothesis)
# metrics_dict = nlgeval.compute_metrics(reference, hypothesis)
# print(metrics_dict)
