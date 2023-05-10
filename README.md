# Sketch2Fashion

This is the post-training utility repository for Sketch2Fashion final year project.

## Steps to run pre-trained model:
### Setup Task
```console
git clone https://github.com/EmmanuelJojy/Sketch2Fashion.git
cd Sketch2Fashion
pip install -r requirements.txt
sh download_weight.sh
```

### Post Setup
To run first add your sketch in the root. The format and filename is okay until you adhere with the same while invoking `sketch2fashion()` method from `transform.py`.

Once the image and argument is resolved run:
```console
python transform.py
```
The script will load the networks with corresponding weights and generate images for each of the models: `out_resnet.jpg`, `out_unet.jpg` and `out_pix2pix.jpg`.