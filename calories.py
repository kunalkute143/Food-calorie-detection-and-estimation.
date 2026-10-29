"""
calories.py
-----------
Approximate calorie data for each food class.

IMPORTANT: These are only APPROXIMATE values. Real calories depend on
portion size, ingredients, cooking method and recipe.

The keys MUST match the folder names inside dataset/train (spelling, lower case).
If you add a new food class, add it to BOTH dictionaries below.
"""

# Calories (kcal) per 100 grams of food
CALORIES_PER_100G = {
    # --- classes from the small Indian food dataset (download_dataset.py) ---
    "burger": 295,
    "fried_rice": 163,
    "idli": 58,
    "jalebi": 390,
    "masala_dosa": 168,
    "pav_bhaji": 140,
    "pizza": 266,
    "samosa": 262,
    # --- extra classes if you use Food-101 (prepare_dataset.py) ---
    "donuts": 452,
    "french_fries": 312,
    "hamburger": 295,
    "ice_cream": 207,
    "pancakes": 227,
}

# A typical serving weight in grams (used when the user does not enter a portion)
TYPICAL_SERVING_GRAMS = {
    "burger": 150,
    "fried_rice": 200,
    "idli": 80,          # about 2 idlis
    "jalebi": 50,
    "masala_dosa": 120,  # one dosa
    "pav_bhaji": 250,    # one plate
    "pizza": 110,        # about one slice
    "samosa": 50,
    "donuts": 60,
    "french_fries": 117,
    "hamburger": 150,
    "ice_cream": 100,
    "pancakes": 150,
}

DISCLAIMER = (
    "Calorie values are approximate. Actual calories depend on portion size, "
    "ingredients, preparation method and recipe."
)


def estimate_calories(food_name, grams=None):
    """
    Estimate calories for a food.

    food_name : predicted class name, e.g. "pizza"
    grams     : portion size in grams (None or 0 -> use typical serving)

    Returns (calories, grams_used) or None if the food is not in the dictionary.
    """
    if food_name not in CALORIES_PER_100G:
        return None
    if grams is None or grams <= 0:
        grams = TYPICAL_SERVING_GRAMS.get(food_name, 100)
    calories = CALORIES_PER_100G[food_name] * grams / 100
    return round(calories), grams


def pretty_name(food_name):
    """'french_fries' -> 'French Fries'"""
    return food_name.replace("_", " ").title()
