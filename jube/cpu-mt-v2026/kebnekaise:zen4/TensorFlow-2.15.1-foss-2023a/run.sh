#!/bin/bash

set -euo pipefail

jube run benchmark.xml -t t1 -o artifacts/t1
jube run benchmark.xml -t t2 -o artifacts/t2
jube run benchmark.xml -t t3 -o artifacts/t3
jube run benchmark.xml -t t4 -o artifacts/t4
jube run benchmark.xml -t t5 -o artifacts/t5
