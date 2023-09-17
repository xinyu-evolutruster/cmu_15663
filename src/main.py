import skimage
from scipy.interpolate import interp2d
import numpy as np
from matplotlib import pyplot as plt
import argparse

black = 150.0
white = 4095.0

r_scale = 2.394351
g_scale = 1.0
b_scale = 1.597656

M_rgb_xyz = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)
M_xyz_cam = np.array(
    [[6988, -1384, -714],
     [-5631, 13410, 2447],
     [-1485, 2204, 7318]]
) / 10000.0


def findPattern(img):
    # Try to find the pattern by comparing adjacent green pixels.
    # Adjacent green pixels are either (upleft, bottomright) or (upright, bottom left).

    # grbg and gbrg
    g1 = img[0::2, 0::2]
    g2 = img[1::2, 1::2]
    # calculate the mean difference between g1 and g2
    mean_diff1 = np.average(np.absolute(g1 - g2))

    # rggb and bggr
    g1 = img[0::2, 1::2]
    g2 = img[1::2, 0::2]
    mean_diff2 = np.average(np.absolute(g1 - g2))

    # GRBG and GBRG: diff = 0.01
    # RGGB and BGGR: diff = 0.005
    # The pattern should be one of RGGB and BGGR
    # Perform White balancing assuming the pattern is RGGB or BGGR,
    # decide which pattern to use based on the result.
    print(
        'GRBG and GBRG: absolute mean difference = {:.6f}'.format(mean_diff1))
    print(
        'RGGB and BGGR: absolute mean difference = {:.6f}'.format(mean_diff2))


def applyPattern(img, pattern='rggb'):
    new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.double)

    # apply green color
    new_img[0::2, 1::2, 1] = img[0::2, 1::2]
    new_img[1::2, 0::2, 1] = img[1::2, 0::2]

    if pattern == 'rggb':
        # red color
        new_img[0::2, 0::2, 0] = img[0::2, 0::2]
        # blue color
        new_img[1::2, 1::2, 2] = img[1::2, 1::2]
    elif pattern == 'bggr':
        # blue color
        new_img[0::2, 0::2, 2] = img[0::2, 0::2]
        # red color
        new_img[1::2, 1::2, 0] = img[1::2, 1::2]

    return new_img


def whiteBalance(img, mode='avg'):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    new_img = np.zeros(img.shape, dtype=np.double)
    if mode == 'avg':    # gray world white balancing
        r_avg = r.mean()
        g_avg = g.mean()
        b_avg = b.mean()

        new_img[:, :, 0] = r * g_avg / r_avg
        new_img[:, :, 1] = g * g_avg / g_avg
        new_img[:, :, 2] = b * g_avg / b_avg
    elif mode == 'max':  # white world white balancing
        r_max = r.max()
        g_max = g.max()
        b_max = b.max()

        new_img[:, :, 0] = r * g_max / r_max
        new_img[:, :, 1] = g
        new_img[:, :, 2] = b * g_max / b_max
    elif mode == 'scaled_avg':
        new_img[:, :, 0] = r * r_scale
        new_img[:, :, 1] = g * g_scale
        new_img[:, :, 2] = b * b_scale
    else:
        print("Error: wrong mode.")

    return new_img


def demosaic(img):

    # Interpolate red
    x_r = np.arange(0, img.shape[1], step=2)
    y_r = np.arange(0, img.shape[0], step=2)
    z_r = img[0::2, 0::2, 0]
    f_red = interp2d(x_r, y_r, z_r, kind='linear')

    # Interpolate blue
    x_b = np.arange(1, img.shape[1], step=2)
    y_b = np.arange(1, img.shape[0], step=2)
    z_b = img[1::2, 1::2, 2]
    f_blue = interp2d(x_b, y_b, z_b, kind='linear')

    # Interpolate green
    x_g1 = np.arange(1, img.shape[1], step=2)
    y_g1 = np.arange(0, img.shape[0], step=2)
    z_g1 = img[0::2, 1::2, 1]
    f_green1 = interp2d(x_g1, y_g1, z_g1, kind='linear')

    x_g2 = np.arange(0, img.shape[1], step=2)
    y_g2 = np.arange(1, img.shape[0], step=2)
    z_g2 = img[1::2, 0::2, 1]
    f_green2 = interp2d(x_g2, y_g2, z_g2, kind='linear')

    # Interpolate red color value for green pixels that are on the same row as red pixels.
    red_val = f_red(x_g1, y_g1)
    img[0::2, 1::2, 0] = red_val
    # Interpolate red color value for green pixels that are on the same column as red pixels.
    red_val = f_red(x_g2, y_g2)
    img[1::2, 0::2, 0] = red_val

    # Interpolate blue color value for green pixels that are on the same column as blue pixels.
    blue_val = f_blue(x_g1, y_g1)
    img[0::2, 1::2, 2] = blue_val
    # Interpolate blue color value for green pixels that are on the same row as blue pixels.
    blue_val = f_blue(x_g2, y_g2)
    img[1::2, 0::2, 2] = blue_val

    # Update green color value
    green_val = f_green1(x_g1, y_g1) + f_green2(x_g1, y_g1)
    img[0::2, 1::2, 1] = green_val
    green_val = f_green2(x_g2, y_g2) + f_green2(x_g2, y_g2)
    img[1::2, 0::2, 1] = green_val

    # interpolate green for red pixels
    green_val = (f_green1(x_r, y_r) + f_green2(x_r, y_r))
    img[0::2, 0::2, 1] = green_val

    # interpolate blue for red pixels
    blue_val = f_blue(x_r, y_r)
    img[0::2, 0::2, 2] = blue_val

    # interpolate red for blue pixels
    red_val = f_red(x_b, y_b)
    img[1::2, 1::2, 0] = red_val

    # interpolate green for blue pixels
    green_val = (f_green1(x_b, y_b) + f_green2(x_b, y_b))
    img[1::2, 1::2, 1] = green_val

    return img


def colorSpaceCorrection(img):
    M_rgb_cam = M_xyz_cam.dot(M_rgb_xyz)
    M_rgb_cam = M_rgb_cam / M_rgb_cam.sum(axis=1).reshape(3, 1)

    M_rgb_cam_inv = np.linalg.inv(M_rgb_cam)

    img_2 = img[..., np.newaxis]
    img_2 = np.matmul(M_rgb_cam_inv, img_2)
    img_2 = img_2.squeeze(axis=3)

    return img_2


def brightnessAdjustment(img, post_intensity=0.5):
    gray_img = skimage.color.rgb2gray(img)
    # print(gray_img.shape)
    # print(gray_img.mean())
    cur_intensity = gray_img.mean()
    brightened_img = img + (post_intensity - cur_intensity)

    return brightened_img


def gammaEncoding(img):
    threshold = 0.0031308

    mask = img < threshold
    img[mask] = 12.92 * img[mask]
    img[~mask] = (1 + 0.055) * np.power(img[~mask], 1 / 2.4) - 0.055

    return img


def manualWhiteBalancing(img):
    plt.imshow(img)
    plt.waitforbuttonpress()
    pts = np.asarray(plt.ginput(n=2, timeout=-1), dtype=np.int32)
    print(pts)
    plt.close()

    subimg = img[pts[0, 1]:pts[1, 1], pts[0, 0]:pts[1, 0]]

    r = subimg[:, :, 0]
    g = subimg[:, :, 1]
    b = subimg[:, :, 2]

    r_avg = r.mean()
    g_avg = g.mean()
    b_avg = b.mean()

    print("r_avg = {}, g_avg = {}, b_avg = {}".format(r_avg, g_avg, b_avg))

    new_img = np.zeros(img.shape)
    new_img[:, :, 0] = img[:, :, 0] * g_avg / r_avg
    new_img[:, :, 1] = img[:, :, 1] * g_avg / g_avg
    new_img[:, :, 2] = img[:, :, 2] * g_avg / b_avg

    return new_img


def show_img(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ImageProcessingPipeline',
    )
    parser.add_argument('-m', '--mode', default='processing',
                        choices=['find_bayer_pattern', 'processing', 'manual_white_balancing'])
    parser.add_argument('-f', '--filename')
    parser.add_argument('-wb', '--white_balancing', type=bool, default=False)
    parser.add_argument('-wm', '--wb_mode', default='max',
                        help='max (white world white balancing), mean (gray world white balancing) or scale')
    parser.add_argument('-d', '--demosaicing', type=bool, default=False)
    parser.add_argument('-c', '--color_correction', type=bool, default=False)
    parser.add_argument('-b', '--brightness', type=bool, default=False)
    parser.add_argument('-pi', '--post_intensity', type=float, default=0.3)
    parser.add_argument('-g', '--gamma_encoding', type=bool, default=False)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # Load image
    file_path = args.filename
    try:
        img = skimage.io.imread(file_path)
    except IOError:
        print("The file {} does not exist.".format(file_path))
        quit()

    print("Image {} is loaded successfully!")
    print("Img height: {}, Img width: {}, every pixel is a {}".format(
        img.shape[0], img.shape[1], img.dtype
    ))

    # Convert image into a double-precision array
    img = img.astype(np.double)

    # Linearization
    scale = white - black
    img = (img - black) / scale
    img = np.clip(img, a_min=0, a_max=1)

    # rggb image
    rggb_img = applyPattern(img, 'rggb')

    if args.mode == 'find_bayer_pattern':
        findPattern(img)
    elif args.mode == 'manual_white_balancing':
        # white balancing
        manual_wb = manualWhiteBalancing(rggb_img)
        plt.imshow(manual_wb * 5)
        plt.show()
    else:  # args.mode == 'processing'
        perform_wb = args.white_balancing
        wb_mode = args.wb_mode
        perform_demosaicing = args.demosaicing
        perform_color_correction = args.color_correction
        perform_brightness_adjustment = args.brightness
        perform_gamma_encoding = args.gamma_encoding

        if perform_wb or perform_demosaicing \
                or perform_color_correction or perform_brightness_adjustment \
                or perform_gamma_encoding:
            if wb_mode == 'max':
                wb_img = whiteBalance(rggb_img, 'max')
            elif wb_mode == 'mean':
                wb_img = whiteBalance(rggb_img, 'avg')
            elif wb_mode == 'scale':
                wb_img = whiteBalance(rggb_img, 'scaled_avg')

            # rggb_img = np.clip((rggb_img), a_min=0, a_max=1)
            if perform_wb:
                show_img(wb_img * 5, 'Result after white balancing')
                # plt.imshow(wb_img * 5)
                # plt.axis('off')
                # plt.savefig('wb_img.png', bbox_inches='tight', pad_inches=0)

        if perform_demosaicing or perform_color_correction \
                or perform_brightness_adjustment or perform_gamma_encoding:

            demosaiced_img = demosaic(wb_img)
            if perform_demosaicing:
                show_img(demosaiced_img, 'Result after demosaicing')

        if perform_color_correction or perform_brightness_adjustment \
                or perform_gamma_encoding:

            # color_correction(img2)
            corrected_img = colorSpaceCorrection(demosaiced_img)
            if perform_color_correction:
                show_img(corrected_img, 'Result after color correction')

        if perform_brightness_adjustment:
            post_intensity = args.post_intensity
            brightened_img = brightnessAdjustment(
                corrected_img, post_intensity)
            show_img(brightened_img, 'Result after brightness adjustment')

        if perform_gamma_encoding:
            gamma_encoded_img = gammaEncoding(corrected_img)
            show_img(gamma_encoded_img, 'Result after gamma encoding')


if __name__ == '__main__':
    main()
