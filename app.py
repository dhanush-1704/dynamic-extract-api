import json
import os
import re
from typing import Any, Dict

from dateutil.parser import parse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, ConfigDict, Field

MODEL = "gemini-3.5-flash-lite"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DynamicRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    schema_: Dict[str, str] = Field(alias="schema")


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    prompt = f"""
You are an expert information extraction engine.

Extract structured information from the text.

TEXT:
{req.text}

TARGET SCHEMA:
{json.dumps(req.schema_, indent=2)}

IMPORTANT RULES:

Return ONLY valid JSON.

Return EXACTLY the keys in the schema.

Do NOT add extra keys.

If a value cannot be determined, return null.

Supported types:

string
integer
float
boolean
date
array[string]
array[integer]

Formatting:

- string -> JSON string
- integer -> JSON integer
- float -> JSON number
- boolean -> true/false
- date -> YYYY-MM-DD
- array[string] -> JSON array
- array[integer] -> JSON array

Never wrap JSON inside markdown.
Never explain anything.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    text = response.text.strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except Exception:
        data = {}

    result = {}

    for field, dtype in req.schema_.items():

        value = data.get(field)

        if value is None:
            result[field] = None
            continue

        try:

            if dtype == "string":
                result[field] = str(value)

            elif dtype == "integer":
                result[field] = int(value)

            elif dtype == "float":
                result[field] = float(value)

            elif dtype == "boolean":

                if isinstance(value, bool):
                    result[field] = value
                else:
                    result[field] = str(value).strip().lower() in (
                        "true",
                        "yes",
                        "1",
                    )

            elif dtype == "date":

                s = str(value).strip()

                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                    result[field] = s
                else:
                    try:
                        result[field] = parse(s).date().isoformat()
                    except Exception:
                        result[field] = s

            elif dtype == "array[string]":

                if isinstance(value, list):
                    result[field] = [str(x) for x in value]
                else:
                    result[field] = [str(value)]

            elif dtype == "array[integer]":

                if isinstance(value, list):
                    result[field] = [int(x) for x in value]
                else:
                    result[field] = [int(value)]

            else:
                result[field] = None

        except Exception:
            result[field] = None

    return result
