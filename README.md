# Supervised Composition and Riff Engine (SC0RE)
Riff generation using various Transformers Architecture

## Requirements
- python > 3.11
- pip

## Running
### Training
1. Put training files inside `gprofiles/{MUSIC_STYLE}`
2. Include the styles in `TRAINING` variable inside `main.py`.
3. Set `MODE` inside `main.py` to `0` for training.
4. Run `main.py`.

### Inference
1. Set `MODE` to `1` inside `main.py` for evaluation.
2. Change `START_ID` to required value in `main.py`.
3. Run `main.py`.