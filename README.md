## requirement
- pip3
- virtualenv

## installation
let's create a virtualenv : 
```bash
python3 -m virtualenv venv
```
then activate the virtualenv
```bash
source venv/bin/activate
```
and then install requirements of the project
```bash
pip install -r requirements.txt
```

## activate virtualenv
```bash
source venv/bin/activate
```

## deactivate virtualenv
```bash
deactivate
```

## train the model
```bash
python src/train.py DATASET_DIRECTORY_PATH
```

## use the model to upscale a video and save it into an .avi file
```bash
python src/upscale_video.py SOURCE_VIDEO_PATH AVI_FILE_PATH
```

## Related works
  Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network by Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang
 BDS500 dataset from Contour Detection and Hierarchical Image Segmentation
 P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
 IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.
