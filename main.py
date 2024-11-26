from dataclasses import dataclass
from pathlib import Path

import spacy
from paddleocr import PaddleOCR

# import paddle
# paddle.utils.run_check()


@dataclass
class BusinessCard:
    raw_text: str
    person: str | None = None
    company: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None


@dataclass
class BusinessCardProcessor:
    ocr: PaddleOCR
    nlp: spacy

    def recognize_person(self, text: str) -> str | None:
        # Process the text
        doc = self.nlp(text)

        # Extract PERSON components
        people_components = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people_components.append(ent.text)
        person = " ".join(people_components)
        return None if not person else person

    def recognize_organization(self, text: str) -> str | None:
        # Process the text
        doc = self.nlp(text)

        # Extract ORG components
        org_components = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_components.append(ent.text)
        org = " ".join(org_components)
        return None if not org else org

    def extract_text_from_image(self, image_path: Path) -> str | None:
        # Perform OCR on an image
        result = self.ocr.ocr(image_path)

        # Aggregate the discovered text
        texts = []
        for line in result:
            for text in line:
                texts.append(text[1][0])

        if len(texts) == 0:
            return None
        return ", ".join(texts)

    def process(self, image_path: Path | str) -> BusinessCard:
        extracted_text = self.extract_text_from_image(str(image_path))
        card = BusinessCard(
            raw_text=extracted_text if extracted_text is not None else ""
        )

        if self.recognize_organization(card.raw_text) != []:
            card.company = self.recognize_organization(card.raw_text)
        if self.recognize_person(card.raw_text) != []:
            card.person = self.recognize_person(card.raw_text)
        return card


def main():
    # Initialize the processor
    processor = BusinessCardProcessor(
        ocr=PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en", show_log=False),
        nlp=spacy.load("en_core_web_trf"),
    )

    for index in range(5):
        card = processor.process(
            Path(f"/Users/dstoker/Downloads/business_card_{index}.png")
        )
        print(f"PERSON: {card.person}")
        print(f"ORG: {card.company}")
        print()

        # Best effort


if __name__ == "__main__":
    main()
