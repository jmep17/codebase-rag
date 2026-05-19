---
id: "fastapi-route"
title: "FastAPI route"
triggers: ["fastapi", "api route", "endpoint"]
file_globs: ["*.py"]
packages: ["fastapi"]
max_tokens: 320
---

Keep route handlers thin: validate input, call application logic, and return typed response models.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Item(BaseModel):
    id: str
    name: str


app = FastAPI()


@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: str) -> Item:
    item = load_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return item
```
