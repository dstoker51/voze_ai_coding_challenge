# import paddle
from dataclasses import dataclass
from pathlib import Path

import spacy
from paddleocr import PaddleOCR

# paddle.utils.run_check()


@dataclass
class Entity:
    person: str
    company: str


def main():
    # Initialize the OCR engine
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en", show_log=False)
    nlp = spacy.load("en_core_web_trf")

    def extract_text_from_image(image_path: Path) -> str:
        # Perform OCR on an image
        result = ocr.ocr(image_path)

        # Aggregate the discovered text
        texts = []
        for line in result:
            for text in line:
                # print(text[1])
                texts.append(text[1][0])

        return ", ".join(texts)

    def recognize_person(text: str) -> str | None:
        # Process the text
        doc = nlp(text)

        # Extract address components
        people_components = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people_components.append(ent.text)
        person = " ".join(people_components)
        return None if not person else person

    def recognize_organization(text: str) -> str | None:
        # Process the text
        doc = nlp(text)

        # Extract address components
        org_components = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_components.append(ent.text)
        org = " ".join(org_components)
        return None if not org else org

    for index in range(5):
        text = extract_text_from_image(
            f"/Users/dstoker/Downloads/business_card_{index}.png"
        )
        person = recognize_person(text)
        org = recognize_organization(text)
        if person and org:
            entity = Entity(person, org)
            print(f"PERSON: {entity.person}")
            print(f"ORG: {entity.company}")
            print()


if __name__ == "__main__":
    main()
