# OCR Check validators

A validator is a Python file with a `validate(text)` function that returns `True` / `False`
(or `(False, "reason")` to explain a failure). The text is already normalised by the class settings
(upper-cased, spaces / dashes / dots removed by default), so validators stay short.

```python
NAME = "Meter reading"
DESCRIPTION = "5 or 6 digits"

def validate(text: str):
    if not text.isdigit():
        return False, "contains non-digits"
    return 5 <= len(text) <= 6
```

Built-in (this folder): ISO 6346 container ID, Indian number plate (incl. BH series), VIN, EAN/UPC/GTIN, ISBN,
IBAN, date, digits only, letters and digits.

Your own validators live in your user folder (shared by all datasets) or in `<dataset>/.aio_qc/validators/`.
In AIO QC: OCR Check ▸ Validators… — create one from a pattern template (`AA 00 A{1,2} 0000`), learn the pattern
from correct examples, or write code; test it live and against the dataset's reads before saving.
A class can use several validators: any one passing is enough (e.g. container ID *or* ISO size code).
