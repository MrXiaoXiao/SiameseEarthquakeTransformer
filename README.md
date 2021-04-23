# SiameseEarthquakeTransformer

## UNDER CONSTRUCTION. The code cleaning and the tutorial update will be finished before 05.08.2021.
A tutorial and updated codes for research paper: ‘Siamese Earthquake Transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic array.’

## Brief Introduction:
Siamese Earthquake Transformer (S-EqT) is developed based on the Earthquake Transformer (EqT) (Mousavi et al., 2020, Nature Communications) (https://github.com/smousavi05/EQTransformer), which is an excellent method and a strong baseline for earthquake detection and phase picking. The primary purpose of the S-EqT model is to reduce the false-negative rate of the EqT model by leveraging latent information in the pre-trained EqT model and retrieving previously missed phase picks in low SNR seismograms based on their similarities with other confident phase picks in high-dimensional spaces.

## Installation

## Citation
If you use the S-EqT codes in your research, please cite both:
Zhuowei Xiao, Jian Wang*, Chang Liu, Juan Li, Liang Zhan, and Zhenxing Yao. (2021). Siamese Earthquake Transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic array. Journal of Geophysics Research: Solid Earth.
and
S. Mostafa Mousavi, William L Ellsworth, Weiqiang Zhu, Lindsay Y Chuang, and Gregory C Beroza. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications 11, 3952. https://doi.org/10.1038/s41467-020-17591-w.

If you use the pipeline in the tutorial, please cite the following papers as well:
REAL for linking seismic phases:
Miao Zhang, William L Ellsworth, and Gregory C Beroza. (2019). Rapid Earthquake Association and Location. Seismological Research Letters, 90(6), 2276–2284. https://doi.org/10.1785/0220190052
HypoInverse for locating earthquakes:
Fred W Klein. (2002). Userʼs Guide to HYPOINVERSE-2000, a Fortran Program to Solve for Earthquake Locations and Magnitudes 4/2002 version. USGS, Open File Report 02-171 Version, 1, 123.

## Bug report
If you occur any bugs or questions, you can either open a new issue in this repo or send me an e-mail (xiaozhuowei@mails.iggcas.ac.cn).
