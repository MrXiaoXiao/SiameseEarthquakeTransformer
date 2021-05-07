# SiameseEarthquakeTransformer
Tutorials and updated codes for the research paper: ‘Siamese Earthquake Transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic array.’
## **The tutorials and codes have been tested on a Linux workstation.** I will constantly update this repo to make them more easy to use and understand.
## Brief Introduction:
Siamese Earthquake Transformer (S-EqT) is developed based on the Earthquake Transformer (EqT) (Mousavi et al., 2020, Nature Communications) (https://github.com/smousavi05/EQTransformer), which is an excellent method and a strong baseline for earthquake detection and phase picking. The primary purpose of the S-EqT model is to reduce the false-negative rate of the EqT model by leveraging latent information in the pre-trained EqT model and retrieving previously missed phase picks in low SNR seismograms based on their similarities with other confident phase picks in high-dimensional spaces.

The S-EqT codes are for building the pre-trained S-EqT model.

The tutorial_01 shows the motivation of the S-EqT model.

The tutorial_02 shows how to use the S-EqT model for building earthquake catalogs from real-world continous seismic data.

## Installation
```Bash
conda create -n seqt
conda activate seqt
conda install python=3.6 tensorflow-gpu=1.14 keras-gpu=2.3.1 h5py=2.10 matplotlib=3.2 pyyaml cudatoolkit cudnn pandas tqdm pyproj jupyter notebook basemap
conda install -c conda-forge obspy
pip install keras-rectified-adam
```
Then enter the directories of tutorials and execute corresponding notebooks and scripts.
## Citation
If you use the S-EqT codes in your research, please cite both:

Zhuowei Xiao, Jian Wang*, Chang Liu, Juan Li, Liang Zhao, and Zhenxing Yao. (2021). Siamese Earthquake Transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic array. Journal of Geophysics Research: Solid Earth. https://doi.org/10.1029/2020JB021444

and

S. Mostafa Mousavi, William L Ellsworth, Weiqiang Zhu, Lindsay Y Chuang, and Gregory C Beroza. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications 11, 3952. https://doi.org/10.1038/s41467-020-17591-w

If you use the pipeline in the tutorial, please cite the following papers as well:

REAL for linking seismic phases:

Miao Zhang, William L Ellsworth, and Gregory C Beroza. (2019). Rapid Earthquake Association and Location. Seismological Research Letters, 90(6), 2276–2284. https://doi.org/10.1785/0220190052

HypoInverse for locating earthquakes:

Fred W Klein. (2002). Userʼs Guide to HYPOINVERSE-2000, a Fortran Program to Solve for Earthquake Locations and Magnitudes 4/2002 version. USGS, Open File Report 02-171 Version, 1, 123.

## Limitations of this work
1. Several questions remain regarding the attributes of feature maps extracted from the EqT model. For example, to what degree the path effect is discarded? How much information do the extracted features carry on event source? 

2. The parameters of the EqT model are fixed in this study. The performance of S-EqT may be improved by applying a loss that constraints latent feature maps to optimize the pre-trained EqT model.

3. Because the feature enhancing module is designed based on the observation of feature maps inside the pre-trained EqT model, it may not generalize well with other models. This module may be removed if the backbone model for feature extraction is trained from the stretch. 

4. The geometry of the template and searching stations, which may benefit the similarity measurement, is not considered in the S-EqT model.

5. The batch size in training and testing is set to be one for implantation convenience, which limits the training and testing speed.

## Bug report
If you occur any bugs or questions, you can either open a new issue in this repo or send me an e-mail (xiaozhuowei@mails.iggcas.ac.cn). 

## Acknowlegments
We would like to thank S. Mostafa Mousavi and his colleagues for developing the EqT model (https://github.com/smousavi05/EQTransformer), which is the base of our S-EqT model.

We would like to thank Miao Zhang for developing REAL (https://github.com/Dal-mzhang/REAL).

We would like to thank Fred Klein for developing HypoInverse (https://www.usgs.gov/software/hypoinverse-earthquake-location)

We would like to thank Yijian Zhou for developing the python interface for HypoInverse (https://github.com/YijianZhou/Hypo-Interface-Py)