# Chromakey-video-to-image

Chromakey-like video and image processing tool that removes backgrounds without requiring an actual green screen. The tool automatically detects the most dominant background color and outputs PNG files with alpha channel transparency.

## Features

- **Process videos**: Extract and process all frames from video files
- **Process images**: Process single images with background removal
- **Automatic background detection**: Finds the most dominant color automatically
- **Multiple color spaces**: Support for both HSV and BGR color space masking
- **Parameter testing**: Built-in tool to find optimal parameters for your content
- **Post-processing**: Optional edge desaturation for cleaner results
- **Multiprocessing**: Fast video processing using all available CPU cores

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Commands

**Process a video:**
```bash
python chromakey_video2png.py --video input.mp4 -o output/
```

**Process an image:**
```bash
python chromakey_video2png.py --image input.png -o output/
```

**Find best parameters for a video:**
```bash
python chromakey_video2png.py --find-best input.mp4 -o test_results/
```

**Find best parameters for an image:**
```bash
python chromakey_video2png.py --find-best-image input.png -o test_results/
```

### Command Line Options

```
Required arguments:
  -v, --video VIDEO           Path to input video file
  -i, --image IMAGE           Path to input image file
  --find-best FIND_BEST       Find best parameters for a video
  --find-best-image           Find best parameters for an image
  -o, --output OUTPUT         Output folder for PNG files

Processing options:
  --hsv                       Use HSV color space (default)
  --bgr                       Use BGR color space instead
  -c, --mask-channel {0,1,2}  Mask channel (default: 1)
                              0=Hue/Blue, 1=Saturation/Green, 2=Value/Red
  -f, --frames FRAMES         Max frames to process for videos (default: -1 = all)
  -m, --margin MARGIN         Crop margin (default: 50)
  -k, --kernel KERNEL         Kernel size for morphology (default: 3)
  -d, --dilate DILATE         Dilation iterations (default: 1)
  -b, --blur BLUR             Median blur size (default: 5)
  --mask-out                  Output mask visualization
  --post-process              Apply edge desaturation
  --post-margin POST_MARGIN   Post-processing margin (default: 20)
```

### Workflow for Best Results

1. **Test parameters**: Run with `--find-best` or `--find-best-image` to generate 6 test outputs with different color space and channel combinations

2. **Choose best result**: Review the generated files (e.g., if `HSV_0.png` looks best, use `--hsv --mask-channel 0`)

3. **Fine-tune settings**: Adjust the following parameters:
   - `--margin` (10-90): Controls how much of similar colors to remove
   - `--kernel` (0-5): Size of morphological operations
   - `--dilate` (0-5): Expands the mask boundaries
   - `--blur` (0-5): Smooths the mask edges

4. **Test on limited frames** (videos only): Use `-f 10` to process only the first 10 frames before committing to full processing

5. **Final processing**: Run with your optimized parameters on the full video/image

### Example

```bash
# Step 1: Find best parameters
python chromakey_video2png.py --find-best video.mp4 -o test/

# Step 2: After reviewing test/, you find HSV channel 0 works best
# Test with limited frames and custom margin
python chromakey_video2png.py --video video.mp4 -o output/ --hsv -c 0 -m 30 -f 10

# Step 3: Process full video with final parameters
python chromakey_video2png.py --video video.mp4 -o output/ --hsv -c 0 -m 30 --post-process
```

## Example Output

<img src="example/output_vid/find_best/00000_HSV_0.png" alt="Example result"/>

## Technical Details

- **Utilizes**: OpenCV for image processing, multiprocessing for parallel video frame processing
- **Output format**: PNG with alpha channel (RGBA)
- **Performance**: Processes video frames in parallel using all available CPU cores minus one

## Credits

Based on [Stack Overflow solution](https://stackoverflow.com/a/66355953)
