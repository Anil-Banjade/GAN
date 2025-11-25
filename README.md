
## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Train
python train.py --epochs 20 --batch-size 64

## Generate Samples
python inference.py --checkpoint ./artifacts/generator.pt --num-samples 10

###  Models aren't trained properly due to resource constraint and hence the results shown in report and in .ipynb files aren't clear. This implementation was primarily focused on exploring and experimenting with the architecture, rather than achieving optimal results. Future work will involve proper training, hyperparameter tuning, and dataset refinement to enhance the quality of generated images and improve the overall iperformance of the model.
