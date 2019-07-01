import argparse
import json

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def eval_sent(input_file,
              output_file):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    with open(output_file, "w") as file:
        correct = 0
        total = 0
        for data in input_data:
            label = data["sent_label"].strip().lower() if "sent_label" in data else data["label"].strip().lower()
            predict = data["sent_predict"].strip().lower() if "sent_predict" in data else data["predict"].strip().lower()
            
            if label == predict:
                correct += 1
            
            total += 1
        
        accuracy = (float(correct) / float(total)) if total > 0 else 0.0
        file.write("Accuracy: {0}".format(accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    eval_sent(args.input_file, args.output_file)
