# cs230
Here's how to run tensorboardX for PyTorch.

1. pip install tensorboardX (https://github.com/lanpa/tensorboardX)
2. pip install tensorflow (if you haven't already)
3. pip install tensorboard (if you haven't already)
4. Run "tensorboard --logdir='.' --port 6006" in the terminal
5. Open up the tensorboard page in chrome/firefox/explorer using port 6006. Example: "http://don-desktop:6006"
6. Run the test\_tb.py "python test\_tb.py"
7. Check your port page and you should see a plot of the losses per iteration. 

