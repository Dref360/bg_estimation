import numpy as np
import scipy.ndimage

MIN_UNIQUE_COLORS = 4096
COLOR_DENSITY_RATIO = 0.11

QUALITY_IN_MIN = 82

ERROR_THRESHOLD = 1.3

ERROR_THRESHOLD_INACCURACY = 0.01


def compute_ssim(im1, im2, l=255):
    # k1,k2 & c1,c2 depend on L (width of color map)
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2

    window = np.ones((8, 8)) / 64.0

    # Convert image matrices to double precision (like in the Matlab version)
    im1 = im1.astype(np.float)
    im2 = im2.astype(np.float)

    # Means obtained by Gaussian filtering of inputs
    mu_1 = scipy.ndimage.filters.convolve(im1, window)
    mu_2 = scipy.ndimage.filters.convolve(im2, window)

    # Squares of means
    mu_1_sq = mu_1 ** 2
    mu_2_sq = mu_2 ** 2
    mu_1_mu_2 = mu_1 * mu_2

    # Squares of input matrices
    im1_sq = im1 ** 2
    im2_sq = im2 ** 2
    im12 = im1 * im2

    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = scipy.ndimage.filters.convolve(im1_sq, window)
    sigma_2_sq = scipy.ndimage.filters.convolve(im2_sq, window)

    # Covariance
    sigma_12 = scipy.ndimage.filters.convolve(im12, window)

    # Centered squares of variances
    sigma_1_sq -= mu_1_sq
    sigma_2_sq -= mu_2_sq
    sigma_12 -= mu_1_mu_2

    if (c_1 > 0) & (c_2 > 0):
        ssim_map = (((2 * mu_1_mu_2 + c_1) * (2 * sigma_12 + c_2)) /
                    ((mu_1_sq + mu_2_sq + c_1) * (sigma_1_sq + sigma_2_sq + c_2)))
    else:
        numerator1 = 2 * mu_1_mu_2 + c_1
        numerator2 = 2 * sigma_12 + c_2

        denominator1 = mu_1_sq + mu_2_sq + c_1
        denominator2 = sigma_1_sq + sigma_2_sq + c_2

        ssim_map = np.ones(mu_1.size)

        index = (denominator1 * denominator2 > 0)

        ssim_map[index] = ((numerator1[index] * numerator2[index]) /
                           (denominator1[index] * denominator2[index]))
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    # return MSSIM
    index = np.mean(ssim_map)

    return index
