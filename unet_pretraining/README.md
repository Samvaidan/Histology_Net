### Color Pretraining and UNET Training

You can replicate model creation by following the example in the color_pretrain.ipynb

You can also directly use the repository provided.
Add data to the /data folder. Create the following directory structure:
- data
    - train
        - labelled
            - images
            - masks
        - unlabelled
    - test
        - labelled
    - validation
        - labelled
            - images
            - masks
        - unlabelled


Create a virtual environment using conda and then install requirements.txt using the command: conda install --yes --file requirements.txt --channel default --channel pytorch --channel nvidia --channel conda-forge
After that run color_pretrain.py to pretrain a unet architecture.
After pretraining run finetune.py to finetune the pretrained unet on the task. Having finetuned, the model can be used for making predictions.