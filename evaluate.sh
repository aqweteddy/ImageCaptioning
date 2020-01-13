mkdir evalute
python evalute.py   --answer_file data/annotations/captions_val2014.json\
                    --model_file model/resnet-lstm/val.txt\
                    --hypothesis evalute/hypothesis_lstm_layer1.txt\
                    --reference evalute/reference_lstm_layer1.txt
nlg-eval --hypothesis=evalute/hypothesis_lstm_layer1.txt --references=evalute/reference_lstm_layer1.txt