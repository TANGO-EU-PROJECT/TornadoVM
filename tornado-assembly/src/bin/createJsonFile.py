#!/usr/bin/env python3

#
# Copyright (c) 2013-2023, APT Group, Department of Computer Science,
# The University of Manchester.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Script to create a Json file from the output generated by TornadoVM when the profiler
is enabled.

Usage: ./createJsonFile.py <inputFile.json> <outpuFile.json>
"""

import json
import sys


## Add here content to the blacklist as needed
def ignore(line):
    if line.startswith("Size"):
        return True
    if line.startswith("Median"):
        return True
    return False


def createOneJsonFile(inputFileName, outputFileName="output.json"):
    f = open(inputFileName, "r")
    lines = f.readlines()
    entryIndex = 0
    jsonContent = {}
    jsonEntry = ""
    for line in lines:
        if line.startswith("{"):
            jsonEntry = line
        elif line.startswith("}"):
            ## json entry complete
            jsonEntry = jsonEntry + line
            entry = json.loads(jsonEntry)
            jsonContent[str(entryIndex)] = entry
            entryIndex = entryIndex + 1
            jsonEntry = ""
        elif not ignore(line):
            jsonEntry = jsonEntry + line

    with open(outputFileName, "w") as outfile:
        json.dump(jsonContent, outfile)


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 2:
        print("Processing file: " + sys.argv[1])
        createOneJsonFile(sys.argv[1])
    elif len(sys.argv) == 3:
        print("Processing file: " + sys.argv[1] + ", outputFile: " + sys.argv[2])
        createOneJsonFile(sys.argv[1], sys.argv[2])
    else:
        print("Usage: ./createJsonFile.py <inputFile.json> <outpuFile.json>")
