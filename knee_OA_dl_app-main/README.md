# Knee Osteoarthritis Analysis with X-ray Images using Deep Convolutional Neural Networks

This code repository is the final project of the Complex Data Mining course at
Unicamp (MDC013).

## Knee Osteoarthritis

[Knee osteoarthritis](https://en.wikipedia.org/wiki/Knee_arthritis) is a
pathology that occurs due to wear on the cartilage that protects the bones in
this region from friction and impacts.

Some medical procedures are necessary to identify this pathology, such as
**X-rays** or magnetic resonance imaging, in which it is possible to assess the
loss in joint spacing, thus indicating the severity of the disease.

The severity of osteoarthritis was classified into 5 levels based on [KL
score](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4925407/), from the healthy
level to the severe level, where the greater the degree of severity, the smaller
the spacing of the joint.



The following image shows the different levels from [Knee Osteoarthritis Dataset
with Severity


## Purpose

The purpose of this project is to correctly classify the severity of
osteoarthritis based on X-ray images.


## Project Structure

```shell
.
├── README.md
├── app
│   ├── app.py
│   └── img
├── assets
├── dataset
│   ├── test
│   ├── train
│   └── val
├── environment.yml
└── src
    ├── 01_data_preparation.ipynb
    ├── 02_ensemble_models.ipynb
    ├── 02_model_inception_resnet_v2.ipynb
    ├── 02_model_resnet50.ipynb
    ├── 02_model_xception.ipynb
    ├── 03_best_model_on_test_xception.ipynb
    └── models
        └── model_Xception_ft.hdf5
```

