
## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Train
python train.py --epochs 20 --batch-size 64

## Generate Samples
python inference.py --checkpoint ./artifacts/generator.pt --num-samples 10

## Model is not trained properly and hence the results aren't clear. It was just done for the purpose of playing with the architecture

