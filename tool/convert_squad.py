import argparse
import json

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--span_file", help="path to answer span file", required=True)
    parser.add_argument("--prob_file", help="path to no-answer probability file", required=True)
    parser.add_argument("--prob_thres", help="threshold of no-answer probability", required=False, default=1.0, type=float)

def convert_squad(input_file,
                  span_file,
                  prob_file,
                  prob_thres):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    span_dict = {}
    prob_dict = {}
    for data in input_data:
        qas_id = data["qas_id"]
        predict_text = data["predict_text"]
        answer_prob = data["answer_prob"] if "answer_prob" in data else 0.0
        
        span_dict[qas_id] = predict_text if answer_prob < prob_thres else ""
        prob_dict[qas_id] = answer_prob
    
    with open(span_file, "w") as file:
        json.dump(span_dict, file, indent=4)
    
    with open(prob_file, "w") as file:
        json.dump(prob_dict, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_squad(args.input_file, args.span_file, args.prob_file, args.prob_thres)
