import sys
import os
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import tiltshift


def readImages(image_dir):
    """This function reads in input images from a image directory

    Args:
    ----------
        image_dir : str
            The image directory to get images from.

    Returns:
    ----------
        images : list
            List of images in image_dir. Each image in the list is of type
            numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png', 'JPG']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    """
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
            .format(image_dir))
    """
    return image_files

def unlinkDirectoryContents(dir):
    """This function deletes all contents from a directory path

    Args:
    ----------
        dir : str
           Path to directory requiring cleanup

    Returns:
    ----------
        N/A

    """
    try:
        for item in os.listdir(dir):
            item_path = os.path.join(dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    cur_dir = os.getcwd()
    image_dir = os.path.join(cur_dir, "input")
    out_dir = os.path.join(cur_dir, "output")

    if len(sys.argv) == 7:
        y1 = float(sys.argv[1])
        y2 = float(sys.argv[2])
        theta = float(sys.argv[3])
        sat = float(sys.argv[4])
        i = int(sys.argv[5])
        image_file = sys.argv[6]
        print("Processing image %d." % (i))
        image = cv2.imread(os.path.join(image_dir, image_file), cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
        tiltshift_image = tiltshift.applyEffect(image, y1, y2, theta, sat, gaussian_or_box='gaussian', save_image=False)
        tiltshift.saveImage(tiltshift_image, "%d.png" % i)
        quit()

    print("Removing contents from out dir: %s." % out_dir)
    unlinkDirectoryContents(out_dir)

    print("Reading images.")
    images = readImages(image_dir)
    image = cv2.imread(images[0], cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)

    # FUTURE: Ensure all images are of the same dimensions

    print("Rendering image and controls.")
    h, w = image.shape[:2]
    center_point = tiltshift.centerPoint(image)
    y2 = int(h / 3) # Bottom line where tilt-shift effect stops
    y1 = 2 * y2 # Top line where tilt-shift effect starts
    theta = 0 # Angle to adjust tilt-shift lines
    y1_coords = [(0, y1), (w, y1)]
    y2_coords = [(0, y2), (w, y2)]

    fig = plt.figure()

    # Show image via MPL
    ax = fig.add_axes([0, 0.3, 1, 0.6]) #[left, bottom, width, height]
    extent = (0, w, 0, h)
    im = ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), origin='upper', extent=extent)

    # Show start and stop lines for tilt-shift effect
    y1_plot, = ax.plot([x for x,y in y1_coords], [y for x,y in y1_coords], linewidth=4, color='blue')
    y2_plot, = ax.plot([x for x,y in y2_coords], [y for x,y in y2_coords], linewidth=4, color='blue')

    # Add sliders to control start and stop lines for tilt-shift effect
    axcolor = 'lightgoldenrodyellow'
    axy1= plt.axes([0.05, 0.15, 0.35, 0.03], facecolor=axcolor)
    sy1 = Slider(axy1, 'Y1', int(h / 2), h - 40, valinit=y1, valstep=1)
    axy2 = plt.axes([0.55, 0.15, 0.35, 0.03], facecolor=axcolor)
    sy2 = Slider(axy2, 'Y2', 40, int(h / 2), valinit=y2, valstep=1)

    # Add slider to control angle of start and stop lines
    axtheta = plt.axes([0.05, 0.10, 0.35, 0.03], facecolor=axcolor)
    stheta = Slider(axtheta, 'Θ', -90, 90, valinit=theta, valstep=1)

    # Add slider to control saturation
    axsat = plt.axes([0.55, 0.10, 0.35, 0.03], facecolor=axcolor)
    ssat = Slider(axsat, 'Sat.', 0, 2, valinit=1, valstep=0.1)

    # Call-back when sliders are updated
    def update(val):
        y1 = sy1.val
        y2 = sy2.val
        theta = stheta.val
        sat = ssat.val

        coords = tiltshift.updateYCoordinates(y1_coords, y1)
        new_y1_coords = tiltshift.rotatePoints(center_point, coords, -1 * theta)
        y1_plot.set_xdata([x for x,y in new_y1_coords])
        y1_plot.set_ydata([y for x,y in new_y1_coords])

        coords = tiltshift.updateYCoordinates(y2_coords, y2)
        new_y2_coords = tiltshift.rotatePoints(center_point, coords, -1 * theta)
        y2_plot.set_xdata([x for x,y in new_y2_coords])
        y2_plot.set_ydata([y for x,y in new_y2_coords])

        update_img = tiltshift.increaseSaturation(image, sat)
        im = ax.imshow(cv2.cvtColor(update_img, cv2.COLOR_BGR2RGB), origin='upper', extent=extent)

        fig.canvas.draw_idle()
    sy1.on_changed(update)
    sy2.on_changed(update)
    stheta.on_changed(update)
    ssat.on_changed(update)

    # Add button to preview tilt-shift effect on first input image
    previewax = plt.axes([0.35, 0.025, 0.1, 0.04])
    previewbutton = Button(previewax, 'Preview', color=axcolor, hovercolor='0.975')

    # Call-back when preview button is clicked
    def preview(event):
        print("In preview.")

        y1 = sy1.val
        y2 = sy2.val
        theta = stheta.val
        sat = ssat.val

        coords = tiltshift.updateYCoordinates(y1_coords, y1)
        new_y1_coords = tiltshift.rotatePoints(center_point, coords, theta)
        new_y1_coords = tiltshift.rotatePoints((w,h), new_y1_coords, -1 * theta)
        y1 = np.max([np.abs(y) for x,y in new_y1_coords])

        coords = tiltshift.updateYCoordinates(y2_coords, y2)
        new_y2_coords = tiltshift.rotatePoints(center_point, coords, theta)
        new_y2_coords = tiltshift.rotatePoints((w,h), new_y2_coords, -1 * theta)
        y2 = np.max([np.abs(y) for x,y in new_y2_coords])

        tiltshift_image = tiltshift.applyEffect(image, y1, y2, theta, sat, gaussian_or_box='gaussian', save_image=True)
        im = ax.imshow(cv2.cvtColor(tiltshift_image, cv2.COLOR_BGR2RGB), origin='upper', extent=extent)
    previewbutton.on_clicked(preview)

    # Add button to compute tilt-shift effect on all input images
    computeax = plt.axes([0.5, 0.025, 0.1, 0.04])
    computebutton = Button(computeax, 'Process', color=axcolor, hovercolor='0.975')

    # Call-back when compute button is clicked
    def compute(event):
        y1 = sy1.val
        y2 = sy2.val
        theta = stheta.val
        sat = ssat.val

        coords = tiltshift.updateYCoordinates(y1_coords, y1)
        new_y1_coords = tiltshift.rotatePoints(center_point, coords, theta)
        new_y1_coords = tiltshift.rotatePoints((w,h), new_y1_coords, -1 * theta)
        y1 = np.max([np.abs(y) for x,y in new_y1_coords])

        coords = tiltshift.updateYCoordinates(y2_coords, y2)
        new_y2_coords = tiltshift.rotatePoints(center_point, coords, theta)
        new_y2_coords = tiltshift.rotatePoints((w,h), new_y2_coords, -1 * theta)
        y2 = np.max([np.abs(y) for x,y in new_y2_coords])

        for i, image_file in enumerate(images):
            print("Processing image %d." % (i))
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
            tiltshift_image = tiltshift.applyEffect(image, y1, y2, theta, sat, gaussian_or_box='gaussian', save_image=False)
            tiltshift.saveImage(tiltshift_image, "%d.png" % i)
    computebutton.on_clicked(compute)

    plt.show()