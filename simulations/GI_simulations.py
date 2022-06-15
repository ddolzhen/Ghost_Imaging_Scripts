

import random
import numpy as np
import numba



def createFrames(nframes,screen_dimensions):
    return np.zeros(shape=(nframes,screen_dimensions[0],screen_dimensions[1]))

@numba.njit(parallel=True)
def addUniformNoise(frames,noise_max):
    for i in numba.prange(frames.shape[0]):
        for j in numba.prange(frames.shape[1]):
            for k in numba.prange(frames.shape[2]):
                frames[i, j, k] += random.randrange(0, noise_max)

@numba.njit(parallel=True)
def addSmallSpecklesNegativeBinomial(frames,r,p):
    for i in numba.prange(frames.shape[0]):
        for j in numba.prange(frames.shape[1]):
            for k in numba.prange(frames.shape[2]):
                frames[i, j, k] += random.randrange(0, np.random.negative_binomial(r,p) )


def specklePatternOld(frames,speckles_per_frame):
    for i in numba.prange(frames.shape[0]):
        for k in numba.prange(speckles_per_frame):
            addSpeckleOld(random.randrange(10, 90), random.randrange(10, 90), random.randrange(0, 8) * 2 + 1, frames,
                       random.randrange(0, 100), i)

# NOTE: Size only takes odd arguments
@numba.njit()
def addSpeckleOld(x, y, size, dataset, intensity_peak, frame):
    x_size = dataset.shape[1]
    y_size = dataset.shape[2]

    if size % 2 == 1:
        speckle_x1 = x - (size - 1) // 2
        speckle_x2 = x + (size - 1) // 2

        speckle_y1 = y - (size - 1) // 2
        speckle_y2 = y + (size - 1) // 2

    else:
        return
    num_stages = (size + 1) / 2

    stage_intensity = intensity_peak / num_stages
    while (speckle_x2 - speckle_x1 != 0):
        dataset[frame, speckle_x1:speckle_x2 + 1, speckle_y1:speckle_y2 + 1] += stage_intensity

        speckle_x1 += 1
        speckle_x2 -= 1
        speckle_y1 += 1
        speckle_y2 -= 1

    dataset[frame, x, y] += stage_intensity


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



def definedMasks(name,frames):
    mask = np.ndarray(shape=(frames.shape[1],frames.shape[2]),
                      dtype=int)  # Has to be same size as speckle dataset. 1 for no object, 0 for object
    mask.fill(1)
    if name=="square":
        mask[40:60, 40:60] = 0
        return mask
    elif name == "wire":
        mask[49:53, 0:100] = 0
        return mask
    elif name == "circle":
        mask=create_circular_mask(frames.shape[1],frames.shape[2],radius=10)
        mask=np.invert(mask)
        return mask

    return 0

@numba.njit(parallel=True)
def cutOutMask(frames,mask,noise_range=0):
    if noise_range !=0:

        anti_mask = np.invert(mask)
        for i in numba.prange(frames.shape[0]):
            for j in numba.prange (frames.shape[1]):
                for k in numba.prange(frames.shape[2]):
                    frames[i,j,k]=frames[i,j,k] * mask [j,k]   + random.randrange(0,noise_range) * anti_mask [j,k]
    else:
        for i in numba.prange(frames.shape[0]):
            for j in numba.prange (frames.shape[1]):
                for k in numba.prange(frames.shape[2]):
                    frames[i,j,k]=frames[i,j,k] * mask [j,k]


@numba.njit(parallel=True)
def ghostPixel(frames,frames_with_object):
    ghost = np.zeros(shape=(frames.shape[1],frames.shape[2]), dtype=np.double)


    nframes=frames.shape[0]
    buckets = []

    for i in range(nframes):
        buckets.append(np.sum(frames_with_object[i, :, :]))

    for i in numba.prange(100):
        for j in numba.prange(100):
            term1 = 0
            term2 = 0
            term3 = 0
            # This loop goes over 10000 frames
            for frame_ct in numba.prange(nframes):
                # This is I_i
                beam1_bucket = np.double(buckets[frame_ct])

                # This is S_i
                beam2_pixel = np.double(frames[frame_ct, i, j])
                # Where i is the index of a frame (frame_ct in code)
                # Note that this 'i' is not the same as  the 'i' in the first for-loop above. This 'i' is the x coordinate of the image

                term1 += beam1_bucket * beam2_pixel
                term2 += beam1_bucket
                term3 += beam2_pixel

            # After the three summations have been assembled, put them into formula (3) to produce the ghost image at the [i,j] coordinate
            ghost[i, j] = term1 / nframes - (term2 * term3) / (nframes ** 2)
    return ghost





if __name__ == "__main__":
    frames=createFrames(10000,(100,100))

    addSmallSpecklesNegativeBinomial(frames,40,0.5)

    frames_with_object=np.copy(frames)

