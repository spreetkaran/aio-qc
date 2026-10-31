from typing import Literal
from typing import TypeAlias

AiOutputFormat: TypeAlias = Literal[
    "rectangle", "polygon", "mask", "circle", "oriented_rectangle"
]
