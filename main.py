import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mysql.connector
import spacy
import typer
from paddleocr import PaddleOCR

SPACY_MODEL = "en_core_web_trf"

# import paddle
# paddle.utils.run_check()

app = typer.Typer()


@dataclass
class BusinessCard:
    raw_text: str
    person: str | None = None
    company: str | None = None
    zip: str | None = None

    def valid(self):
        return (
            self.person is not None
            and self.company is not None
            and self.zip is not None
        )

    def info(self):
        print(self.raw_text)
        print()
        print(f"Name: {self.person}")
        print(f"Company: {self.company}")
        print(f"Zip: {self.zip}")
        print()
        print()


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

        # State
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


def load_spacy_model(model: str):
    model_path = os.path.join("./models/spacy", model)
    try:
        nlp = spacy.load(model_path)
    except OSError:
        print(f"{model} not found at {model_path}. Downloading...")
        spacy.cli.download(model)
        nlp = spacy.load(model)
        nlp.to_disk(model_path)
    return nlp


@app.command()
def process_card(image_path: Path):
    processor = BusinessCardProcessor(
        ocr=PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en", show_log=False),
        nlp=load_spacy_model(SPACY_MODEL),
    )

    card = processor.process(image_path)

    # Prompt user to correct info as needed
    card.info()
    if not typer.confirm("Does the information look correct?", default=True):
        card.person = typer.prompt("Name?", card.person)
        card.company = typer.prompt("Company?", card.company)
        card.zip = typer.prompt("Zip Code?", card.zip)
        card.info()

    # Connect to DB
    mydb = mysql.connector.connect(
        host="sql3.freesqldatabase.com",
        user="sql3743596",
        password="rwP5Nr2azB",
        database="sql3743596",
    )
    # print(mydb)

    mycursor = mydb.cursor()

    sql = "SELECT * FROM Contacts WHERE FirstName ='Sarah'"

    mycursor.execute(sql)

    myresult = mycursor.fetchall()

    for x in myresult:
        print(x)


if __name__ == "__main__":
    app()
