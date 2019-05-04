# Fake Miniatures

## Synopsis

Miniature faking is a process in which a photograph of a life-size scene is made to look like a photograph of a miniature scale model. This codebase is a limited Python reproduction of Adobe Photoshop’s Tilt-Shift Tool. In addition, I incorporated a filter to increase the color saturation of an image to make it appear more "toy-like."

A five-step computational pipeline is implemented:

1. Input (1 or more images + parameters)
2. Adjust Color Saturation (Automatic)
3. Decrease Depth of Field (Automatic)
4. Animate Images (Manual with FFmpeg)
5. Output (timelapsed video)

An example of the tool at work is shown below:
![Traffic Circle](trafficcircle.png?raw=true)

## Pre-requisites
1. Python
2. OpenCV 4.0.0.21
3. numpy 1.15.2
4. matplotlib 3.0.0
5. imutils 0.5.2

## To run
1. Place all input images in the ./input folder. All images should be of the same dimensions.
2. Run `python main.py`
3. After applying the effect to all input images, the output is saved in the ./output folder.
4. Combine output images into a timelapse video:
`ffmpeg -r 24 -f image2 -s widthxheight -i %d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p out.mp4`

## Example Output
An example output is available online at:
https://drive.google.com/file/d/1UhZnqZMA42vqShnRwrCJSOHDsKLWK_vM/view?usp=sharing