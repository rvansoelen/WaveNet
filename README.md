# WaveNet
A tensorflow implementation of DeepMind's WaveNet

Data was obtained from the YouTube 8M dataset. The dataset is partitioned into {frame_level,video_level} and {train,validate,test}. Only frame level data was used.

To obtain the dataset, do:
  curl data.yt8m.org/download.py | partition=1/video_level/train mirror=us python

Or to download just 1/1000th of the data:
  curl data.yt8m.org/download.py | shard=1,1000 partition=1/video_level/train mirror=us python