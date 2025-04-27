import cv2
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import filedialog
from spellchecker import SpellChecker
from scipy import ndimage
import re

# Set the Tesseract executable path (modify if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Skew correction using projection profile
def correct_skew(image, delta=1, limit=45):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def determine_score(arr, angle):
        data = ndimage.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return score

    angles = np.arange(-limit, limit + delta, delta)
    scores = [(determine_score(thresh, angle), angle) for angle in angles]
    best_score, best_angle = max(scores)

    rotated = ndimage.rotate(image, best_angle, reshape=False)
    return rotated

# Preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return None

    # Resize image
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    denoised = cv2.medianBlur(gray, 5)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Skew correction
    edges = cv2.Canny(sharpened, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angle = 0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle_deg = (theta * 180 / np.pi) - 90
            if -45 < angle_deg < 45:
                angles.append(angle_deg)
        if angles:
            angle = np.median(angles)

    (h, w) = sharpened.shape
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(sharpened, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Thresholding
    try:
        _, binary_img = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except:
        binary_img = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    return binary_img

# OCR using pytesseract
def extract_text_from_image(image_path):
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None

    # Optional: Show processed image
    # cv2.imshow("Processed Image for OCR", processed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    try:
        text = pytesseract.image_to_string(processed_img, lang='eng')
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return None

# Spell checker
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected = [spell.correction(word) or word for word in words]
    return ' '.join(corrected)

# Check if text is messy
def is_text_messy(text):
    messy_chars = re.findall(r'[^a-zA-Z0-9\s\.,]', text)
    if len(messy_chars) > 5:
        return True
    if len(text) < 20:
        return False
    return False

# Fuzzy character-level correction (optional)
def fuzzy_character_correction(text):
    replacements = {'O': '0', 'o': '0', 'l': '1', 'I': '1'}
    corrected = []
    for word in text.split():
        if word.isdigit() or any(char.isdigit() for char in word):
            # Correct words that seem like numbers
            new_word = ''.join([replacements.get(ch, ch) for ch in word])
            corrected.append(new_word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

# Final processing function
def process_text(text):
    print("\n--- Original OCR Text ---")
    print(text)

    # Step 1: Spell Check
    corrected_text = correct_spelling(text)
    print("\n--- After Spell Checking ---")
    print(corrected_text)

    # Step 2: Conditional Fuzzy Correction
    if is_text_messy(text):
        fuzzy_corrected_text = fuzzy_character_correction(corrected_text)
        print("\n--- After Fuzzy Character Correction (Applied) ---")
    else:
        fuzzy_corrected_text = corrected_text
        print("\n--- After Fuzzy Character Correction (Skipped) ---")
    print(fuzzy_corrected_text)

    return fuzzy_corrected_text

# File picker
def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

# Main
def main():
    image_path = select_image_file()

    if not image_path:
        print("No file selected. Exiting...")
        return

    extracted_text = extract_text_from_image(image_path)
    if extracted_text is None:
        print("Text extraction failed.")
        return

    final_text = process_text(extracted_text)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    print("\n✅ Text has been saved to 'output.txt'.")

if __name__ == "__main__":
    main()
