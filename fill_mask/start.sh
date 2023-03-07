#!/bin/bash 

python3 -m uvicorn main:app --reload --port 8081 --host=0.0.0.0
