import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

'''
Data downloaded from http://www.cs.toronto.edu/~kriz/cifar.html.
'''

class Dataset():

    def __init__(self, data_dir, select_classes=None, seed=0):
        self.data_dir = Path(data_dir)
        self.select_classes = select_classes

        assert self.select_classes == None or len(self.select_classes) == 2

        self.rng = np.random.default_rng(seed)

        self.train_data, self.val_data, self.test_data = self.make_split()


    def make_split(self):
        """
        From the data loaded, create a train/val/test split.
        """
        # shuffle train data
        all_train_data = self.load_train_data()
        m = all_train_data['data'].shape[0]

        shuffle = self.rng.permutation(m)

        all_train_data['data'] = all_train_data['data'][shuffle]
        all_train_data['labels'] = all_train_data['labels'][shuffle]

        # choose 80% as train and 20% as val
        cutoff = int(m * 0.8)
        train_data, val_data = {}, {}

        train_data['data'] = all_train_data['data'][:cutoff, :]
        train_data['labels'] = all_train_data['labels'][:cutoff]

        val_data['data'] = all_train_data['data'][cutoff:, :]
        val_data['labels'] = all_train_data['labels'][cutoff:]

        # keep test batch as test data
        test_data = self.load_test_data()

        # preprocess
        all_data = np.concatenate([all_train_data['data'], test_data['data']], axis=0)
        all_mean, all_std = np.mean(all_data), np.std(all_data) # did this so we could get mean/std of all data, both train and test
        
        train_data = self.preprocess(train_data, mean=all_mean, std=all_std)
        val_data = self.preprocess(val_data, mean=all_mean, std=all_std)
        test_data = self.preprocess(test_data, mean=all_mean, std=all_std)

        print(f"Created train/test/val split. train: {train_data['data'].shape}, val: {val_data['data'].shape}, test: {test_data['data'].shape}")

        return train_data, val_data, test_data
    

    def load_train_data(self):
        """
        Load CIFAR-10's prepared batches and concatenate them all.
        """
        data_dict = {} # dict to store data loaded

        for b in range(1, 6):
            
            # open 1 of 6 data batch files
            selected_data, selected_labels = self.load_from_pickle(self.data_dir / f"data_batch_{b}")
            
            if b == 1:
                # start with these arrays (so that concatenation can be done later)
                data_dict['data'] = selected_data
                data_dict['labels'] = selected_labels
            
            else:
                # concat with all our train data
                data_dict['data'] = np.append(data_dict['data'], selected_data, axis=0)
                data_dict['labels'] = np.append(data_dict['labels'], selected_labels, axis=0)

            print(f"Shapes are now: data={data_dict['data'].shape}, labels={data_dict['labels'].shape}")

        return data_dict
    
    
    def load_test_data(self):
        """
        Load CIFAR-10's prepared test batch.
        """
        data_dict = {} # dict to store data loaded
            
        # open test batch file
        selected_data, selected_labels = self.load_from_pickle(self.data_dir / f"test_batch")
            
        data_dict['data'] = selected_data
        data_dict['labels'] = selected_labels
    
        print(f"Shapes are now: data={data_dict['data'].shape}, labels={data_dict['labels'].shape}")

        return data_dict
    
    
    def load_from_pickle(self, data_path):
        """
        Open a pickle file, load the data, and preprocess.

        Preprocess data by:
        - Masking to only include the 2 classes selected by select_classes
        """

        with open(data_path, 'rb') as f:
            batch_dict = pickle.load(f, encoding='bytes')

        batch_labels = np.array(batch_dict[b'labels'])
        
        if self.select_classes:
            # create mask for indexes where label is one of selected labels
            cond = False
            for select_class in self.select_classes:
                cond = cond | (batch_labels == select_class)
            select_mask = np.where(cond)[0]
            
            # mask data
            selected_data = np.array(batch_dict[b'data'])[select_mask, :] # get data
            selected_labels = np.array(batch_dict[b'labels'])[select_mask] # get labels
        else:
            selected_data = np.array(batch_dict[b'data'])
            selected_labels = np.array(batch_dict[b'labels'])

        print(f"Loaded {data_path}, items selected: {len(selected_labels)}")

        return selected_data, selected_labels


    def preprocess(self, data_dict, mean, std):
        """
        Preprocess data by:
        - Averaging channels to create a grayscale image
        - Scaling all image data to the range [-1, 1]
        - Mapping each of the (should be two for this experiment) selected classes to -1 or 1
        """
        
        m, d = data_dict['data'].shape # number of things
        assert d % 3 == 0

        # average channels
        data_by_channels = np.reshape(data_dict['data'], (m, d // 3, 3))
        # print(data_by_channels.shape)
        data_dict['data'] = np.mean(data_by_channels, axis=-1)
        # print(data_dict['data'].shape)

        # scale to [-1, 1] by normalizing
        data_dict['data'] = (data_dict['data'] - mean) / std

        if self.select_classes:
            # map classes to -1 or 1
            data_dict['labels'] = np.where(data_dict['labels'] == self.select_classes[0], -1, 1)

        return data_dict



def make_dataset():
    return Dataset("C:\\Users\\shem\\Desktop\\Fall 2023\\CMPUT 466\\Miniproject\\20170308hundehalter.csv")

if __name__ == "__main__":
    a = Dataset("./Miniproject/cifar-10-batches-py", select_classes=[0, 1], seed=0)