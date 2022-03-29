AVNet in Python using Keras
===============================================================

Overview
------------
Many retinal diseases effect Arteries and Veins in different ways. Clinical observations of the retinal vascularture in diabetic retinopathy patients, suggest a dilation of the veins and a constriction of the arteries. Therefore, the differentiation of arteries and veins can provide important information for diagnosis and monitoring of retinopathies. To examine the retinal vasculature, optical coherence tomography angiography (OCTA) is used. Current methods for artery-vein (AV) pixel-wise classification or segmentation in OCTA requires an extensive image processing procedure, which may inhibit clinical deployment.

In this project we present a deep learning approach for dispersion compensation using a fully convolutional network (FCN)
named Artery-vein classification network (AV-Net) for optical coherence tomography (OCT) and OCT-angiography (OCTA) and evaluate multimodal fusion strategies to improve segmentation performance.

Dataset
------------
The dataset were collected by the Biomedical Optics and Ophthalmic Imaging Laboratory at the University of Illinois at Chicago. This study has been conducted in compliance with the ethical regulations reported in the Declaration of Helsinki and has been authorized by the institutional review board of the University of Illinois at Chicago.

![The input into AVNet is a combination of both en face OCT and OCTA, whereas the output is an RGB image of the AV map.](https://github.com/dleninja/multimodal-avnet/blob/main/misc/example_data.png?raw=true)

Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All en face OCT/OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used. The en face OCT was generated as a 3D projection of the retina slab. After image reconstruction, both en face OCT and OCTA were exported from ReVue software interface (Optovue) for further processing.

Network Architecture
------------
The AV-Net is a FCN based on a modified UNet algorithm, which consists of an encoder-decoder architecture. The input to the AV-Net can be of a single channel or a two channel image. The network architecture presented below represents a late fusion approach that combines the outputs of two networks trained on different imaging modalities, OCT and OCTA, respectively.

![The late stage fusion approach of AV-Net, which employs different expert networks for OCT and OCTA, and combines the output of the two networks.](https://github.com/dleninja/multimodal-avnet/blob/main/misc/figure_Late_fusion.png?raw=true)

Dependencies
------------
- tensorflow >= 1.31.1
- keras >= 2.2.4
- python >= 3.7.1

Citations
------------
Abtahi Mansour, David Le, Lim Jennifer, and Xincheng Yao.
"MF-AV-Net: an open source deep learning network with multimodal fusion options for artery-vein segmentation in OCT angiography (2022)".
