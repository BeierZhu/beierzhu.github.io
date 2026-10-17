#!/usr/bin/env bash
set -euo pipefail

bundle install --jobs 4 --retry 2
bash bin/jekyll build
