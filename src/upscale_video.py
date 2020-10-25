import cv2
from tensorflow.keras.models import load_model
import sys
import numpy as np
from tqdm import trange
import constants
import utils

def upscale_image(model, img):
    img_scaled = utils.scaling(img)
    input = np.expand_dims(img_scaled, axis=0)
    out = model.predict(input)

    out_img = out[0] * 255
    out_img.clip(0, 255)
    return out_img

def upscale_video(source_video, model):
    video = cv2.VideoCapture(source_video)
    if (video.isOpened()== False): 
        print("Error opening video stream or file")
    else:
        framerate = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_array = np.ndarray((frame_count, frame_height, frame_width,3), dtype="uint8")
        upscaled_frame_array = np.ndarray((frame_count, frame_height*constants.UPSCALE_FACTOR, frame_width * constants.UPSCALE_FACTOR,3), dtype="uint8")
        for index in trange(frame_count): 
            frame_exists, frame = video.read()
            if (frame_exists):
                frame_array[index] = frame
                upscaled_frame_array[index] = upscale_image(model, frame)
    return upscaled_frame_array, framerate, frame_width, frame_height

def write_video(frame_array, framerate, upscaled_width, upscaled_height, filename):
    out_video = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (upscaled_width,upscaled_height))
    for frame in frame_array:
        out_video.write(frame)
    

def run():
    if (len(sys.argv) != 3):
        print("help : python src/upscale_face.py SOURCE_IMAGE_PATH DESTINATION_FILE_PATH")
    else:
        upscale_model = load_model(constants.MODEL_NAME, custom_objects={"PSNR" : utils.PSNR})
        upscaled_frame_array, framerate, frame_width, frame_height = upscale_video(sys.argv[1],upscale_model)
        write_video(upscaled_frame_array,framerate, frame_width*constants.UPSCALE_FACTOR, frame_height*constants.UPSCALE_FACTOR, sys.argv[2])

run()