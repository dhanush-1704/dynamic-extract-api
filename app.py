import json
import os
import re
from datetime import date
from typing import Dict, Any

from dateutil.parser import parse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

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
    text: str
    schema: Dict[str, str]


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    prompt = f"""
You are an information extraction engine.

Extract ONLY the requested fields.

TEXT:
{req.text}

SCHEMA:
{json.dumps(req.schema, indent=2)}

Rules:

- Return ONLY valid JSON.
- Return EXACTLY the keys in the schema.
- No extra keys.
- No missing keys.
- If value cannot be found return null.
- Dates MUST be YYYY-MM-DD.
- integer -> JSON integer
- float -> JSON number
- boolean -> true/false
- array[string] -> JSON array of strings
- array[integer] -> JSON array of integers
- Never wrap JSON in markdown.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    text = response.text.strip()

    text = re.sub(r"^```json", "", text, flags=re.I).strip()
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

    for field, dtype in req.schema.items():

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
                    result[field] = str(value).lower() == "true"

            elif dtype == "date":
                result[field] = parse(str(value)).date().isoformat()

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
