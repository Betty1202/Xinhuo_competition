# Image Caption

## Environment
Python libraries used:
```shell
torch==1.9.0
torchvision==0.10.0
imageio==2.9.0
h5py==2.10.0
ntlk=3.7
tqdm=4.62.3
scikit-image==0.15.0
```
## Dataset
dataset_download: `Flickr8k-9dea07ba660a722ae1008c4c8afdd303b6f6e53b.torrent`

then run `create_input_files.py` to generate:
```shell
TEST_CAPLENS_flickr8k_5_cap_per_img_5_min_word_freq.json
TEST_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json
TEST_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
TRAIN_CAPLENS_flickr8k_5_cap_per_img_5_min_word_freq.json
TRAIN_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json
TRAIN_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
VAL_CAPLENS_flickr8k_5_cap_per_img_5_min_word_freq.json
VAL_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json
VAL_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json
```
## Training
in the folder `json_hdf5` and run `train.py` to train
