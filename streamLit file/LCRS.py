import cv2
import imutils
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import streamlit as st

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

def detect_license_plate(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = False
    else:
        detected = True

    if detected:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = gray[topx:bottomx+1, topy:bottomy+1]
        return cropped
    else:
        return None

def recognize_license_plate(cropped):
    if cropped is not None:
        text = pytesseract.image_to_string(cropped, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return text
    else:
        return "License plate not detected"

def main():
    st.title("License Plate Recognition App")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        cropped = detect_license_plate(image_path)
        if cropped is not None:
            license_plate_number = recognize_license_plate(cropped)
            st.image(image_path, caption="Uploaded Image", use_column_width=True)
            st.subheader("Detected License Plate Number:")
            st.write(license_plate_number)
        else:
            st.write("License plate not found in the image.")

if __name__ == "__main__":
    main()
