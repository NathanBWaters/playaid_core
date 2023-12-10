'''
Character Predictor loader dataset class
'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import cv2 as cv
# import imgaug.augmenters as iaa

from playaid.constants import CHAR_DETECTOR_DIR, CHAR_INPUT_SIZE
from playaid.data import get_fox_marth_games


CHAR_DATAFRAME = os.path.join(CHAR_DETECTOR_DIR, 'char_dataframe.pkl')


def games_to_char_dataframe(games=None):
    '''
    Convert games into a character dataframe
    '''
    if not games:
        games = get_fox_marth_games()

    data = {'frame_path': [], 'label': []}
    for game in games:
        label = game.char_label()

        for frame_path in game.frame_paths:
            data['frame_path'].append(frame_path)
            data['label'].append(label)

    dataframe = pd.DataFrame(data=data)
    return dataframe


def get_char_dataframe():
    '''
    Retrieves the char dataframe
    '''
    if os.path.exists(CHAR_DATAFRAME):
        return pd.read_pickle(CHAR_DATAFRAME)

    char_dataframe = games_to_char_dataframe()
    char_dataframe.to_pickle(CHAR_DATAFRAME)
    return char_dataframe


def crop_stock_info(frame):
    '''
    To determine which characters are playing we only need the bottom part of
    the screen.  See assets/stock_info.jpg for an example
    '''
    frame = cv.resize(frame, (CHAR_INPUT_SIZE[0], 250))
    frame = frame[-CHAR_INPUT_SIZE[1]:]
    return frame


class CharacterLoader(Dataset):
    '''
    Generates an image of a single Arabic character.
    '''
    def __init__(self,
                 dataframe=get_char_dataframe(),
                 augment=True,
                 transforms=transforms.ToTensor()):
        '''
        Args:
            transforms (callable, optional): Optional transforms to be applied
                on a sample.
        '''
        self.char_dataframe = dataframe
        self.augment = augment
        self.transforms = transforms

    def __len__(self):
        '''
        __len__
        '''
        return len(self.char_dataframe)

    def __getitem__(self, idx):
        '''
        __getitem__
        '''
        row = self.char_dataframe.sample(n=1)
        frame_path = row.frame_path.values[0]
        label = row.label.values[0]
        feature = cv.imread(frame_path)
        feature = crop_stock_info(feature)

        if self.transforms:
            feature = self.transforms(feature)

        return feature, label


if __name__ == '__main__':
    loader = CharacterLoader()
    for i in range(len(loader)):
        feature, label = loader[i]
        print('Good')
        if feature.shape != torch.Size([3, CHAR_INPUT_SIZE[1], CHAR_INPUT_SIZE[0]]):
            import pdb; pdb.set_trace()
            print('Hmmm')
