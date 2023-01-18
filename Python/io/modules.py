import dask
import numpy as np

from tune.io import PZFFormat


@dask.delayed
def decompress(pzf_filepath):
    with open(pzf_filepath, 'rb') as f:
        datastring = f.read()

    data, header = PZFFormat.loads(datastring)
    return data


def merge_lines(block, weighted_pixels, option="merge", start_index=None, crop_length=None, line_no=4):
    if option == "merge":
        img_cropped = block[:, :, start_index:start_index + crop_length, :]
        return np.sum(np.expand_dims(weighted_pixels.T, (0, 3)) * img_cropped, 2, keepdims=True)
    elif option == "single":
        return block[:, :, line_no:line_no + 1, :]


def reposition_lines(data, sample_positions, grid):
    sample_positions = np.squeeze(sample_positions)
    data = np.squeeze(data)
    grid_img = np.zeros_like(data)
    x_length = sample_positions.shape[0]
    sample_idx = 0

    if sample_positions[-1] < sample_positions[0]:
        sample_positions = np.flip(sample_positions)
        data = np.flip(data, 1)

    for i in range(x_length):

        if grid[i] < sample_positions[0]:
            grid_img[:, i] = data[:, 0]
        elif grid[i] > sample_positions[-1]:
            grid_img[:, i] = data[:, -1]
        else:
            while not (sample_positions[sample_idx] <= grid[i] <= sample_positions[sample_idx + 1]):
                sample_idx += 1

            right = sample_positions[sample_idx + 1] - grid[i]
            left = grid[i] - sample_positions[sample_idx]
            grid_img[:, i] = ((right * data[:, sample_idx]) + (left * data[:, sample_idx + 1])) / (right + left)

    return np.expand_dims(grid_img, 0)




