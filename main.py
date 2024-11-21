# import paddle
import spacy
from paddleocr import PaddleOCR

# paddle.utils.run_check()

# Initialize the OCR engine
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en", show_log=False)

# Perform OCR on an image
result = ocr.ocr("/Users/dstoker/Downloads/business_card_1.png")

# Print the results
texts = []
for line in result:
    for text in line:
        print(text[1])
        texts.append(text[1][0])


def recognize_address(text: str):
    # Load a spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Text to process
    text = "I live at 123 Main Street, New York, NY 10001."

    # Process the text
    doc = nlp(text)

    # Extract address components
    address_components = []
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  # Geopolitical Entity or Location
            address_components.append(ent.text)

    # Join the components to form the address
    address = " ".join(address_components)

    print(address)


recognize_address(" ".join(texts))

# import os

# import cv2
# from matplotlib import pyplot as plt
# from paddleocr import PaddleOCR, draw_ocr

# # Initialize the OCR model
# ocr_model = PaddleOCR(
#     lang="en", use_gpu=False
# )  # You can enable GPU by setting use_gpu=True

# # Specify the path to the image you want to perform OCR on
# img_path = "/Users/dstoker/Downloads/business_card_0.png"

# # Perform OCR on the image
# result = ocr_model.ocr(img_path)

# # Get the detected boxes, texts, and scores
# boxes = result[0][0][0]
# text, score = result[0][0][1]

# # Path to the font file for visualization
# font_path = os.path.join("PaddleOCR", "doc", "fonts", "latin.ttf")

# # Load the image using OpenCV and reorder the color channels
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Visualize the image with detected text
# plt.figure(figsize=(15, 15))
# annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)
# plt.imshow(annotated)
# plt.axis("off")  # Turn off axis labels
# plt.show()
