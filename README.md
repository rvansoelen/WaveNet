# WaveNet
A tensorflow implementation of DeepMind's WaveNet

Data was obtained from a collection of youtube videos of piano music. To download the audio of the videos, run:
youtube-dl --extract-audio --audio-format wav [url]

Data can also be obtained from the YouTube 8M dataset. The dataset is partitioned into {frame_level,video_level} and {train,validate,test}. Only frame level data was used.

To obtain the dataset, do (Requires 1.7 TB of free space):
  curl data.yt8m.org/download.py | partition=1/frame_level/train mirror=us python

Or to download just 1/1000th of the data:
  curl data.yt8m.org/download.py | shard=1,1000 partition=1/frame_level/train mirror=us python
