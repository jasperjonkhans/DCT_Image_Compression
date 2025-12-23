import cv2
import numpy as np
from DCT_Image_Compression.compression_format import SimpleComp
import os

PATH = "./test_img.png"

LUMINANCE_QUANTIZATION_TABLE = np.array([
    [16,11,10,16,24 ,40 ,51 ,61 ],
    [12,12,14,19,26 ,58 ,60 ,55 ],
    [14,13,16,24,40 ,57 ,69 ,56 ],
    [14,17,22,29,51 ,87 ,80 ,62 ],
    [18,22,37,56,68 ,109,103,77 ],
    [24,35,55,64,81 ,104,113,92 ],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99 ]
], dtype=np.int16)

CHROMINANCE_QUANTIZATION_TABLE = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.int16)

def test_resize(img):
    return cv2.resize(img.astype(np.float32), (128, 128))

def print_img(channel, channel_description = ""):
    print(f"shape: {channel.shape}")
    print(f"dtype of pixels:  {channel.dtype}") 
    cv2.imshow(channel_description, channel.astype(np.uint8))
    print("press key to close window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def chroma_subsampling(channel):
    return channel[::2, ::2]

def padding(channel, mutliple_of):
    h, w = channel.shape
    if h % mutliple_of != 0 or w % mutliple_of != 0:
        channel = cv2.resize(channel.astype(np.float32), (w//mutliple_of * mutliple_of, h//mutliple_of * mutliple_of)).astype(np.int16)
    return channel

def image_preprocessing(image):
    YCrCb = load_YCrCb(image)
    YCrCb = [padding(x, 16) for x in YCrCb]
    YCrCb = [np.clip(Y_value - 128, 0, 255) for x in YCrCb]
    YCrCb[1] = chroma_subsampling(YCrCb[1])
    YCrCb[2] = chroma_subsampling(YCrCb[2])
    return YCrCb

def load_YCrCb (BGRImage):
    if BGRImage is None:
        raise FileNotFoundError(f"image is None")
    YCrCb = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2YCrCb)
    return cv2.split(YCrCb.astype(np.int16))

def group_processing_forward(input, output, luminance):
    qt = LUMINANCE_QUANTIZATION_TABLE if luminance else CHROMINANCE_QUANTIZATION_TABLE

    dct_coeffs = cv2.dct(input.astype(np.float32))
    output[:] = np.round(dct_coeffs / qt).astype(np.int16)

def group_processing_backward(input, output, luminance):
    qt = LUMINANCE_QUANTIZATION_TABLE if luminance else CHROMINANCE_QUANTIZATION_TABLE
    dequantized = input.astype(np.float32) * qt
    idct_coeffs = cv2.idct(dequantized)
    output[:] = np.round(idct_coeffs).astype(np.int16)

def DCT_transform(u, v, group):
    def alpha(k: int) -> float:
        return 1 / np.sqrt(2) if k == 0 else 1.0
    s = 0.0
    for x in range(8):
        for y in range(8):
            s += group[x, y] * np.cos((2*x + 1) * u * np.pi / 16.0) * np.cos((2*y + 1) * v * np.pi / 16.0)
    return 0.25 * alpha(u) * alpha(v) * s

def to_coeff_rep(Image):
    img_coeff_rep = []
    for channel_idx, channel in enumerate(Image):
        h, w = channel.shape
        channel_coeff_rep = np.empty((h, w), dtype=np.int16)
        for i in range(h // 8):
            for j in range(w // 8):
                slice_i, slice_j = slice(i * 8, i * 8 + 8), slice(j * 8, j * 8 + 8)
                group_processing_forward(channel[slice_i, slice_j], channel_coeff_rep[slice_i, slice_j], channel_idx == 0)
        img_coeff_rep.append(channel_coeff_rep)
    return img_coeff_rep

def to_value_rep(img_coeff):
    img_value_rep = []
    for channel_idx, channel in enumerate(img_coeff):
        h, w = channel.shape
        channel_value_rep = np.empty((h, w), dtype=np.int16)
        for i in range(h // 8):
            for j in range(w // 8):
                slice_i, slice_j = slice(i * 8, i * 8 + 8), slice(j * 8, j * 8 + 8)
                group_processing_backward(channel[slice_i, slice_j], channel_value_rep[slice_i, slice_j], channel_idx == 0)
        img_value_rep.append(channel_value_rep)
    return img_value_rep

if __name__ == "__main__":
    CompressionFormat = SimpleComp
    BGRImage = cv2.imread(PATH)
    img = image_preprocessing(BGRImage)
    Y, Cr, Cb = img
    np.save("original", np.array(img, dtype=object))

    img_coeff = to_coeff_rep(tuple((Y, Cr, Cb)))
    Y_coeff, Cr_coeff, Cb_coeff = img_coeff
    img_value = to_value_rep(img_coeff)
    Y_value, Cr_value, Cb_value = img_value

    SimpleComp.save("compressed_image", img_coeff)

    if False:
        print_img(Y, "Y channel")
        print_img(Cr, "Cr channel")
        print_img(Cb, "Cb channel")

        print_img(Y_coeff)
        print_img(Cb_coeff)
        print_img(Cr_coeff)

        print_img(Y_value)
        print_img(Cb_value)
        print_img(Cr_value)

    h, w = Y_value.shape
    Cr_resized = cv2.resize(Cr_value.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    Cb_resized = cv2.resize(Cb_value.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    Y_rec = np.clip(Y_value + 128, 0, 255).astype(np.uint8)
    Cr_rec = np.clip(Cr_resized + 128, 0, 255).astype(np.uint8)
    Cb_rec = np.clip(Cb_resized + 128, 0, 255).astype(np.uint8)

    merged = cv2.merge([Y_rec, Cr_rec, Cb_rec])
    bgr_img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    original_size = os.path.getsize("original.npy")
    compressed_size = os.path.getsize("compressed_image.npy")
    print(f"Original Size: {original_size / 1024:.2f} KB")
    print(f"Compressed Size: {compressed_size / 1024:.2f} KB")
    print(f"Compression Ratio: {original_size / compressed_size:.2f}")

    cv2.imshow("Decompressed Image", bgr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



