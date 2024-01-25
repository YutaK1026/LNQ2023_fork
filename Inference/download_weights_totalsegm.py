import totalsegmentator
import sys
import os

path = path = os.path.dirname(totalsegmentator.__file__)
sys.path.append(path)

from libs import download_pretrained_weights

if __name__ == "__main__":
    """
    Download all pretrained weights
    """
    download_pretrained_weights(256)
