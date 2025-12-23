# DCT Image Compression

A Python implementation of image compression using Discrete Cosine Transform (DCT) and OpenCV. To measure compression, I saved both of the pictures as np arrays and looked at the file size. 

This is an educational project, currently I am using the cv2 inbuild dct and idct functions because my naive python rebuild is too slow, but the project will be further extended with my own fast-DCT implementation in C. 

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Place your image as `test_img.png` in the project directory and run:

```bash
python main.py
```

## Process

1. **Compression**: Padding → Shift → DCT-II → Quantization
2. **Decompression**: Quantization → DCT-III → Shift

## License

MIT 