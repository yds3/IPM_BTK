# Beyond Textual Knowledge: Leveraging Multimodal Knowledge Bases for Enhancing Vision-and-Language Navigation

This repository is the official implementation of [Beyond Textual Knowledge: Leveraging Multimodal Knowledge Bases for Enhancing Vision-and-Language Navigation]. 





![framework](1/1.png)


## Requirements

1. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator). We use the latest version instead of v0.1.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name vlnduet python=3.8.5
conda activate vlnduet
pip install -r requirements.txt
```

3. Download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0), including processed annotations, features and pretrained models of REVERIE and R2R datasets. Put the data in `datasets' directory.Download the image features extracted by CLIP and place them in the location specified by the code from from [here](https://huggingface.co/crystal61/VLN-GOAT/blob/main/R2R/features/CLIP-ViT-B-16-views.hdf5).Download the multimodal knowledge base (Knowledge) and place it under the datasets directory; meanwhile, download the text instructions extracted by Qwen (annotations_r2r/reverie), and replace the original JSON files in the annotations folders of R2R and REVERIE with them, from [here](https://huggingface.co/datasets/hf258/IPM_BTK).

4. Download pretrained lxmert
```
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```

## Pretraining

Combine behavior cloning and auxiliary proxy tasks in pretraining:
```pretrain
cd pretrain_src
bash run_reverie.sh # (run_r2r.sh)
```

## Fine-tuning & Evaluation

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd map_nav_src
bash scripts/run_reverie.sh # (run_r2r.sh)
```
