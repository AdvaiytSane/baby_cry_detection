# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import timeit
import glob
import warnings

from baby_cry_detection.rpi_methods import Reader
from baby_cry_detection.rpi_methods.feature_engineer import FeatureEngineer
from baby_cry_detection.rpi_methods.majority_voter import MajorityVoter

from baby_cry_detection.rpi_methods.baby_cry_predictor import BabyCryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--load_path_model',
                        default='{}/../../output/model/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../output/prediction/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--log_path',
                        default='{}/../../'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

     # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_prediction_test_test_model.log'),
                        filemode='w',
                        level=logging.INFO)

    all_data = glob.glob("/Users/hsane/aduprojects/baby_cry_detection/data/*/*")

    for file_name in all_data:
        logging.info('Reading {0}'.format(file_name))

        # Read signal (first 5 sec)
        file_reader = Reader(file_name)
        play_list, sr = file_reader.read_audio_file()

        logging.info('Sampling rate {sr}'.format(sr=sr))
        logging.info('Starting feature engineering')
        # Feature extraction
        engineer = FeatureEngineer()
        play_list_processed = list()

        for signal in play_list:
            tmp = engineer.feature_engineer(signal)
            play_list_processed.append(tmp)

        # MAKE PREDICTION
        logging.info('Predicting...')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

        with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
            model = pickle.load(fp)

        predictor = BabyCryPredictor(model)

        predictions = list()
        predicted_labels = list()

        for signal in play_list_processed:
            pred_label = model.predict(signal)
            predicted_labels.append(pred_label)
            tmp = predictor.classify(signal)
            predictions.append(tmp)
            logging.info("Predicted {a} : {b}".format(a=pred_label[0], b= tmp))
        
        # MAJORITY VOTE

        majority_voter = MajorityVoter(predictions)
        majority_vote = majority_voter.vote()

        # SAVE

        logging.info('Saving prediction...')

        # Save prediction result
        with open(os.path.join(save_path, 'prediction.txt'), 'w') as text_file:
            text_file.write("{}".format(majority_vote))

        logging.info('Saved! {}'.format(os.path.join(save_path, 'prediction.txt')))


if __name__ == '__main__':
    main()
