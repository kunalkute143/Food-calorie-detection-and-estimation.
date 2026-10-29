# Food Calorie Detection and Estimation Using CNN

A simple deep-learning application that identifies a food item from a photograph and estimates its approximate calories. A Convolutional Neural Network (CNN) trained with TensorFlow/Keras recognises the food, and the calories are calculated from a nutrition lookup table and the portion size.

This is a Bachelor of Engineering (Information Technology) project, Saraswati College of Engineering, Kharghar (University of Mumbai), academic year 2026-27.

## Features

- Recognises 8 foods from an uploaded image
- Shows the predicted food, the confidence (%) and the estimated calories (kcal)
- Optional portion size in grams for a better estimate
- Probability bar chart for all 8 classes
- Command-line prediction script
- Optional React + Flask interface with a non-food rejection filter

## Supported foods

| | | | |
|---|---|---|---|
| Burger | Fried rice | Idli | Jalebi |
| Masala dosa | Pav bhaji | Pizza | Samosa |

## How it works

1. The image is resized to 128 x 128 and the pixel values are scaled from 0-255 to 0-1.
2. A CNN (3 x Conv2D + MaxPooling2D, Flatten, Dense, Dropout, Softmax) predicts a probability for each of the 8 foods.
3. The highest probability gives the food name, and that probability is shown as the confidence.
4. Calories are calculated as:

   **Calories (kcal) = kcal per 100 g x portion in grams / 100**

   If no portion is entered, a typical serving size is used.

Example: pizza has about 266 kcal per 100 g, so a 250 g portion is about 665 kcal.

## Model summary

| Item | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Input size | 128 x 128 x 3 |
| Classes | 8 |
| Data | about 240 training and 60 validation images per class |
| Augmentation | random flip, rotation, zoom (training only) |
| Optimiser / loss | Adam / sparse categorical cross-entropy |
| Epochs / batch size | 15 / 32 |
| Validation accuracy | 63.43% |

## Project structure

```
food-calorie-detection/
├── train.py            # builds, trains and saves the CNN
├── predict.py          # command-line prediction
├── calories.py         # calorie dictionary and calculation
├── app.py              # Streamlit web app
├── class_names.json    # order of the food classes
├── food_model.h5       # trained model
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.9 - 3.11.

```bash
git clone https://github.com/[your-username]/food-calorie-detection.git
cd food-calorie-detection
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository. Images came from a public Indian-food dataset on Hugging Face [add the dataset name/link and check its licence]. Eight classes were selected and arranged like this:

```
dataset/
├── train/        (about 240 images per class)
│   └── burger/  fried_rice/  idli/  jalebi/  masala_dosa/  pav_bhaji/  pizza/  samosa/
└── validation/   (about 60 images per class)
    └── burger/  fried_rice/  idli/  jalebi/  masala_dosa/  pav_bhaji/  pizza/  samosa/
```

## Usage

**Train the model** (about 5-20 minutes on a normal laptop CPU):

```bash
python train.py
```

**Predict from the command line** (the portion in grams is optional):

```bash
python predict.py test_images/pizza1.jpg 250
```

**Run the web app:**

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Limitations

- Only 8 foods are supported. Any other food is forced into one of these classes.
- Calorie values are approximate, because real calories depend on portion size, ingredients and recipe.
- The CNN is trained from scratch on a small dataset, so its accuracy is modest compared with large pretrained models.
- One food per image; the portion size must be entered by the user.

## Future work

- Transfer learning (e.g. MobileNet) for higher accuracy
- More food classes and more images
- A "not_food" class
- Automatic portion-size estimation and multiple foods per plate
- Protein, fat and carbohydrate estimates
- Mobile app with live camera input

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- Calorie values are approximate and based on nutrition tables such as the *Indian Food Composition Tables 2017* (ICMR-National Institute of Nutrition).
- Built with open-source tools: Python, TensorFlow/Keras, NumPy, Pandas, OpenCV, Matplotlib and Streamlit.
- Developed with the assistance of an AI tool. [Keep or remove this line according to your college's policy and what is true for your team.]
- Project guide: Prof. Rajashree Gavhane.

## Authors

Kunal Kute, Harshad Kolambe, Danish Naik, Pranjal Mokashi

## License

Released under the [MIT License](LICENSE).
