![Banner.jpg](Banner.jpg)
# Supervised Composition and Riff Engine (SC0RE)
Riff generation using various Transformers Architecture

## Requirements
- python > 3.11
- pip

## Running
### Training
1. Put training files inside `gprofiles/{MUSIC_STYLE}`
2. Include the styles in `TRAINING` variable inside `config.py`.
3. Set `MODE` inside `config.py` to `0` for training.
4. Run `main.py`.

### Inference
1. Set `MODE` inside `config.py` to `1` for evaluation.
2. Change `START_ID` to required value in `config.py`.
3. Run `main.py`.