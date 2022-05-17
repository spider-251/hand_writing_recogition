import argparse
import json
from typing import Tuple, List

import cv2
import numpy as np
import os
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())



def infer(model,fn_img):
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return recognized[0], probability[0]


def process_image(file_name):
    # os.makedirs('process_temp',exist_ok=True)

    fname = file_name.split(os.sep)[-1]
    print(fname)
    img = cv2.imread(file_name) 
    # rsz_img = cv2.resize(img, None, fx=0.25, fy=0.25) # resize since image is huge
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    gray = cv2.fastNlMeansDenoising(gray, h=30) 
    # thresh_gray = np.where(gray.ravel() < 127, 0, 255).reshape(gray.shape[0],gray.shape[1])
    thresh_gray = np.where(gray.ravel() < 150, 0, 255)
    for i,j in enumerate(thresh_gray):
        if j == 0:
            for k in range(4):
                thresh_gray[i-k] =0
    thresh_gray = thresh_gray.reshape(gray.shape[0],gray.shape[1])
    cv2.imwrite(os.path.join('static','temp','img.jpg'),thresh_gray)

    # img = cv2.imread('C:\\MAIN_DRIVE\\Projects\\handwriting_recognition\\flaskapp\\outside_test\\davis.jpg')
    img = cv2.imread('static\\temp\\img.jpg')
    
    # Preprocessing the image starts
    
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    xm = []
    ym = []
    wm = []
    hm = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        xm.append(x)
        ym.append(y)
        wm.append(w)
        hm.append(h)
    # rect = cv2.rectangle(im2, (min(xm), min(ym)), (max(xm)+min(wm), max(ym)+max(hm)), (0, 0, 0), 1)
    cropped = im2[min(ym):max(ym) + max(hm), min(xm):max(xm) + max(wm)]

    cv2.imwrite(os.path.join('static','temp','img.jpg'),cropped)

# def parse_args() -> argparse.Namespace:
#     """Parses arguments from the command line."""
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
#     parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
#     parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
#     parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
#     parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
#     parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
#     parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
#     parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
#     parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

#     return parser.parse_args()


def main(path):
    """Main function."""

    # parse arguments and set CTC decoder
    # args = parse_args()
    # decoder_mapping = {'bestpath': DecoderType.BestPath,
    #                    'beamsearch': DecoderType.BeamSearch,
    #                    'wordbeamsearch': DecoderType.WordBeamSearch}
    # decoder_type = decoder_mapping[args.decoder]
    decoder_type = DecoderType.BestPath

    # train the model
    # if args.mode == 'train':
    #     loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

    #     # when in line mode, take care to have a whitespace in the char list
    #     char_list = loader.char_list
    #     if args.line_mode and ' ' not in char_list:
    #         char_list = [' '] + char_list

    #     # save characters and words
    #     with open(FilePaths.fn_char_list, 'w') as f:
    #         f.write(''.join(char_list))

    #     with open(FilePaths.fn_corpus, 'w') as f:
    #         f.write(' '.join(loader.train_words + loader.validation_words))

    #     model = Model(char_list, decoder_type)
    #     train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # # evaluate it on the validation set
    # elif args.mode == 'validate':
    #     loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
    #     model = Model(char_list_from_file(), decoder_type, must_restore=True)
    #     validate(model, loader, args.line_mode)

    # infer text on test image
    # elif args.mode == 'infer':
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump= False)
    rec,prob = infer(model, path)
    return rec,prob
