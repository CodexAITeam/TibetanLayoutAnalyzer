# TibetanLayoutAnalyzer
TibetanLayoutAnalyzer is a tool for analyzing and detecting layouts in Tibetan documents. It includes functionality for generating synthetic training data and training an object detector for Tibetan text blocks.


# Training data
Synthetic training data for Tibetan number detection is generated with `generate_training_data.py` of the `TibetanOCR` submodule. Make sure to update folders for background images.

```python
python ext/TibetanOCR/generate_training_data.py --corpora_folder "data/tibetan numbers/corpora" --background_train "data/tibetan numbers/bg_train" --background_val "data/tibetan numbers/bg_val" --font_path="ext/Microsoft Himalaya.ttf"
```

⚠️ Make sure to set the path to **[Microsoft Himalaya.ttf](https://legionfonts.com/fonts/microsoft-himalaya)** correctly as this font renders Tibetan correctly.


### Command-line Arguments

The script supports various command-line arguments to customize the data generation process:

- `--background_train`: Folder with background images for training (default: './ext/TibetanOCR/data/background_images_train/')
- `--background_val`: Folder with background images for validation (default: './ext/TibetanOCR/data/background_images_val/')
- `--dataset_folder`: Folder for the generated YOLO dataset (default: './data/yolo_tibetan/')
- `--corpora_folder`: Folder with Tibetan tibetan numbers corpora (default: './data/corpora/UVA Tibetan Spoken Corpus/')
- `--train_samples`: Number of training samples to generate (default: 2)
- `--val_samples`: Number of validation samples to generate (default: 1)
- `--no_cols`: Number of text columns to generate [1....5] (default: 1)
- `--font_path`: Path to a font file that supports Tibetan characters (default: 'ext/Microsoft Himalaya.ttf')
- `--single_label`: Use a single label "tibetan" for all files instead of using filenames as labels (flag, no value required)


## Training the Object Detector
After generating the synthetic data, you can train the YOLOv11 object detector: Training of YOLOv11n is done by a CLI call to [Ultralytics](https://docs.ultralytics.com/usage/cli/#train). 

```bash
yolo detect train data=data/yolo_tibetan/tibetan_yolo.yml epochs=1000 imgsz=1024
```

**Note**: You may need to adjust the path field in `tibetan_yolo.yml` to match your directory structure and Ultralytics configuration.

### Model Export and Inference

After training, export the model to TorchScript format:

```bash
yolo detect export model=runs/detect/train/weights/best.pt format=torchscript
```

We can now employ our trained model for inference on new images:

```bash
yolo predict task=detect model=runs/detect/train/weights/best.torchscript imgsz=1024 source=path/to/your/images/*.jpg
```

Results will be saved in the runs/detect/predict directory.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# License
tbd.