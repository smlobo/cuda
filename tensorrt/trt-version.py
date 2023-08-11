#!/usr/bin/env python3

import tensorrt

print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())