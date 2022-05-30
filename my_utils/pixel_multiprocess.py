import os
import numpy as np
import concurrent.futures
from tqdm import tqdm


def _chunk_function(pix_function, pixels_chunk, *args, **kwargs):
    result = []
    for pix in pixels_chunk:
        temp = pix_function(pix, *args, **kwargs)
        result.extend(temp)
    return result


def pixel_multiprocess(pixels, pix_function, *args, **kwargs):
    """
    description
    -----------
    multiprocessing for (see 'note' for details)
    [pix_function(pix, *args, **kwargs) for pix in pixels]

    note
    ----
    pix_function : must return a list: 
        1) [pix_1obj] or 2) [pix2_obj1, pix2_obj2, ...]
    return value would be without sublists:
        [pix_1obj, pix2_obj1, pix2_obj2, ...]
    """
    # setup
    n_cores = int(np.floor(os.cpu_count() * 0.75))  # number of cores used

    n = np.max([int(np.floor(len(pixels) / (10 * n_cores))), 1])
    pixels_partition = [pixels[i:i + n] for i in range(0, len(pixels), n)]
    # multiprocessing
    res_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # res_list = [executor.submit(
        #     _chunk_function, pix_function, pixels_chunk, *args, **kwargs)
        #     for pixels_chunk in tqdm(pixels_partition)]

        # tqdm does not work here
        # for pixels_chunk in tqdm(pixels_partition):
        for pixels_chunk in pixels_partition:
            temp = executor.submit(
                _chunk_function, pix_function, pixels_chunk, *args, **kwargs)
            res_list.append(temp)
        # get result
        final_result = []
        for res in res_list:
            temp = res.result()
            # print(temp)
            final_result.extend(temp)
    return final_result
