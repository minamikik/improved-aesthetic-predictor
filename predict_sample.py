import os
import argparse
from aesthetic_predictor import AestheticPredictor

args = argparse.ArgumentParser()
args.add_argument("--source_dir", type=str, default=".", help="path to the image dir")
args.add_argument("--model_path", type=str, default="models/sac+logos+ava1-l14-linearMSE.pth", help="path to the model")
opt = args.parse_args()

def create_source_list(source_dir):
    source_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                path = os.path.join(root, file)
                source_list.append(path)
    return source_list

def main(source_list):
    print("Predicting the aesthetic score for the following images:")
    for source in source_list:
        predictor = AestheticPredictor(opt.model_path)
        result = predictor.predict(source)
        print(source)
        print(result)

if __name__ == "__main__":
    source_list = create_source_list(opt.source_dir)
    if not len(source_list) == 0:
        main(source_list)
    else:
        print("No images found in the source directory")
