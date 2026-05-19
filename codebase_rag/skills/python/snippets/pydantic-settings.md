---
id: "pydantic-settings"
title: "Pydantic settings"
triggers: ["pydantic", "settings", "config"]
file_globs: ["*.py"]
packages: ["pydantic", "pydantic-settings"]
max_tokens: 300
---

Keep configuration explicit and injectable. Avoid reading environment variables deep inside business logic.

```python
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    database_url: str = Field(default="sqlite:///app.db")
    debug: bool = False


def create_app(config: AppConfig) -> object:
    return {"config": config}
```
