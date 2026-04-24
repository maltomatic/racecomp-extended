# RaceComp
Comparison of CNN and ViT for image upscaling and facial recognition tasks trained on racially biased datasets

## CNNs

### ```cnn-unet.py```
This is a full-U UNet super resolution model with ResNet50 as the encoder backbone. It expects an input tensor of 3x112x12, and upscales to a tensor 3x224x224. Between layers, the decoder upscaling may be center-cropped to match the encoder size, as implemented in the original UNet paper (but here decoder outputs are larger than encoders therefore it is the decoder that is cropped).

### ```cnn_abridged-unet.py```
This is a half-U UNet super resolution model with ResNet50 as the encoder backbone. The top half of the U is not bridged to avoid cropping (zero-pad is used in the lower levels). 

## Data structure
The FairFace dataset should be implemented in the structure shown in ```training_data_structure.jpg```. Reference ```load.py``` to see how the data is loaded and what the resulting data structure may be like. ```data_testbed.ipynb``` allows for interactive viewing and testing of loading data.

## Facial Recognition
First download the flattened RFW dataset from this Google Drive folder: https://drive.google.com/drive/folders/17kOk3Me2C05oUcS_Siarxw89OBGtCbYY?usp=drive_link

Alternatively, if using an original, unmodified RFW download, unzip it in your datasets folder. Then, run rfw_extractor.py. _Modify your `src_image_path` and `dest_image_path` accordingly_. The labels require manual filtering; repeat filenames have been fixed and sorted in rfw_cleaned_labels.csv in the datasets folder. Afterwards, run rfw_cleaner.py to set up the training and testing folders and label csv files. The cleaner file will confirm if all images have labels and if all labels have images, and vice versa. You can also then run load_rfw.py to confirm if the classes and images match up.

In the end, the folder structure should look like this:

```
RaceComp
 |--dataset
     |--RFW
        |--train
        |--val
     --rfw_train_labels.csv
     --rfw_test_labels.csv
```

Additionally, if/when restoring from a checkpoint while training for a minority, please verify that the training classes for the checkpoint are the same as before. With the seed set in ```load_rfw.py``` this should be a given, but it is better to be certain.

# Citations

### **RFW: Racial Faces in the Wild**, **FairFace**, **FastViT**

```bibtex
@inproceedings{wang2019racial,
  title={Racial Faces in the Wild: Reducing Racial Bias by Information Maximization Adaptation Network},
  author={Wang, Mei and Deng, Weihong},
  booktitle={ICCV},
  year={2019}
}

@inproceedings{wang2019caucasian,
  title={Caucasian Face Recognition Dataset and Benchmark},
  author={Wang, Mei and Deng, Weihong},
  booktitle={arXiv preprint arXiv:1812.00194},
  year={2019}
}

@inproceedings{wang2019indian,
  title={Indian Face Recognition Dataset and Benchmark},
  author={Wang, Mei and Deng, Weihong},
  booktitle={arXiv preprint arXiv:1812.00193},
  year={2019}
}

@inproceedings{wang2019asian,
  title={Asian Face Recognition Dataset and Benchmark},
  author={Wang, Mei and Deng, Weihong},
  booktitle={arXiv preprint arXiv:1812.00195},
  year={2019}
}

@misc{karkkainen2021fairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age},
  author={Kimmo Kärkkäinen and Jungseock Joo},
  year={2021},
  eprint={1908.04913},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{vasu2023fastvit,
  title={FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
  author={Vasu, Pranav and Gabriel, John and Zhu, Jun and Jampani, Varun and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={ICCV},
  year={2023}
}
