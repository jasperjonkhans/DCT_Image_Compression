import os
import numpy as np

class SimpleComp:
    @classmethod
    def load(cls, path: str):
        pass
    @classmethod
    def save(cls, path: str, image_coeff_rep: list[np.ndarray]):
        compressed: list = [None] * 3
        for channel_idx, channel in enumerate(image_coeff_rep):
            h_blocks = channel.shape[0] // 8
            w_blocks = channel.shape[1] // 8

            all_vals = []
            all_idxs = []
            counts = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    flat = channel[i*8:(i+1)*8, j*8:(j+1)*8].ravel()
                    idx  = np.flatnonzero(flat).astype(np.uint8)
                    vals = flat[idx].astype(np.int16)

                    all_vals.append(vals)
                    all_idxs.append(idx)
                    counts.append(len(vals))

            if all_vals:
                vals_arr = np.concatenate(all_vals)
                idx_arr = np.concatenate(all_idxs)
            else:
                vals_arr = np.array([], dtype=np.int16)
                idx_arr = np.array([], dtype=np.uint8)
            
            counts_arr = np.array(counts, dtype=np.uint8)

            compressed[channel_idx] = (vals_arr, idx_arr, counts_arr)
        np.save(path, np.array(compressed, dtype=object))