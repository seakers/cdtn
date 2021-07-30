from simulator.parsers.DtnNodeParser import DtnNodeParser
from pydantic import validator
from typing import Any, Dict, List, Optional

class DtnNodeRLParser(DtnNodeParser):
    """ Parser for DtnNode's YAML configuration parameters """
    # Router type. It must be tag of an element defined in the YAML
    maximum_capacity: float = float('inf')

    @validator('maximum_capacity')
    def validate_maximum_capacity(cls, maximum_capacity, *, values, **kwargs):
        return DtnNodeRLParser._validate_tag_exitance(cls, maximum_capacity, values)


