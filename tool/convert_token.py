import argparse
import json

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def convert_token(input_file,
                  output_file):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    with open(output_file, "w") as file:
        for data in input_data:
            text = data["text"]
            label = data["token_label"] if "token_label" in data else data["label"]
            predict = data["token_predict"] if "token_predict" in data else data["predict"]
            text_list = text.strip().split(' ')
            label_list = label.strip().split(' ')
            predict_list = predict.strip().split(' ')
            triple_list = zip(text_list, label_list, predict_list)
            for text, label, predict in triple_list:
                file.write("{0} {1} {2}\n".format(text, label, predict))
            
            file.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_token(args.input_file, args.output_file)
