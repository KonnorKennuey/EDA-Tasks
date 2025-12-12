#!/bin/bash

PAYLOAD='{
  "carat": 0.23,
  "cut": "Ideal",
  "color": "I",
  "clarity": "SI2",
  "depth": 61.5,
  "table": 55,
  "x": 3.95,
  "y": 3.98,
  "z": 2.43
}'

curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  "http://127.0.0.1/api/predict"