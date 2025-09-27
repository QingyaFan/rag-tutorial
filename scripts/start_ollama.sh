#!/usr/bin/env bash

set -e

ollama serve &
ollama run qwen3:0.6b