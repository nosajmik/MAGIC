#!/usr/bin/python3.7
import re
import glog as log
from typing import List

AddrNotFound = -1
FakeCalleeAddr = -2


def findAddrInOperators(operators: List[str]) -> int:
    hexPattern = re.compile(r'$[1-9A-Fa-f][0-9A-Fa-f]*$')
    for item in operators:
        for part in item.split('_'):
            if hexPattern.match(part) is not None:
                log.info(f'"{part}" in {operators} is convertiable to hex int')
                return int(part, 16)
            else:
                log.debug(f'"{part}" is NOT convertiable to hex int')

    return AddrNotFound
