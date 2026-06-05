# subclass classification script

This script takes the "bad/other" (class 3) images from the main classification pipeline and further sub-classifies them into three subcategories:

| # | Subclass     | Description                                                  |
|---|--------------|--------------------------------------------------------------|
| 1 | colored      | Colored or non-transparent plastic bottle (blue, green, red, opaque, translucent, etc.) |
| 2 | dirty        | Dirty or contaminated plastic bottle (filled with liquid/food, residue, grime, mold)   |
| 3 | not_bottle   | NOT a plastic bottle at all (other objects, trash, garbage, non-bottle items)          |

This helps balance the number of images across the three subclasses inside class 3, making it easier to train a CNN model.

## Usage

1. Copy or move the class 3 ("bad") images into the input directory.
2. Set your OpenRouter API key and the input/output paths in `classify_subclass.yaml`.
3. Run:

```bash
cd scripts/classify-subclass
python classify_subclass.py -c classify_subclass.yaml
```

## Requirements

- Python 3.7+
- `requests`, `pyyaml` (install with `pip install requests pyyaml`)
- OpenRouter API key with access to a vision-capable model