import re
from collections import defaultdict
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
    zip: str | None = None


# class BusinessCard(TypedDict):
#     raw_text: str
#     person: str
#     company: str
#     city: str
#     state: str
#     country: str


@dataclass
class BusinessCardProcessor:
    ocr: PaddleOCR
    nlp: spacy

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

    def parse(self, text: str, entity_types: list[str]) -> dict:
        # Process the text
        doc = self.nlp(text)

        # Extract components
        data = defaultdict(list)
        for ent in doc.ents:
            if ent.label_ in entity_types:
                data[ent.label_].append(ent.text)

        return data

    def process(self, image_path: Path | str) -> BusinessCard:
        extracted_text = self.extract_text_from_image(str(image_path))
        discovered_entities = self.parse(extracted_text, ["ORG", "PERSON", "GPE"])
        card = BusinessCard(
            raw_text=extracted_text if extracted_text is not None else "",
            company=" ".join(discovered_entities["ORG"]),
            person=" ".join(discovered_entities["PERSON"]),
        )

        # Extract location information
        # TODO: More than this
        # location_entities = discovered_entities["GPE"]
        # states = list(
        #     set(location_entities).intersection(set(abbreviation_to_name.keys()))
        # )
        # if len(states) > 0:
        #     card.state = states[0]

        # Zip code
        zip_code_pattern = "[0-9]{5}(?:-[0-9]{4})?"
        # TODO: Find a regex pattern that is resistant to other characters instead of
        #       manually filtering first
        zips = re.findall(
            zip_code_pattern,
            " ".join(
                [
                    "".join([val for val in i if val.isalnum() or val in ["-", "."]])
                    for i in extracted_text.split()
                ]
            ),
        )
        # If multiple zips are found snag the first one
        if len(zips) > 0:
            card.zip = zips[0]

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
        print(card.raw_text)
        print()
        print(f"Name: {card.person}")
        print(f"Company: {card.company}")
        print(f"City: {card.city}")
        print(f"State: {card.state}")
        print(f"Country: {card.country}")
        print(f"Zip: {card.zip}")
        print()
        print()

        # print(card)

        # Best effort


if __name__ == "__main__":
    main()
