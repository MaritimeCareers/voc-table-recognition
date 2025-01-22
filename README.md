# VOC Tables Recognition Project

This project aims to process and extract structured data from digitized handwritten tables in the records of the Dutch East India Company (VOC). These records are essential for tracing individual employment histories in the service of the VOC during the eighteenth century.

## Overview

The project focuses on training a machine learning model to recognize and segment table structures from scanned documents. The structured data extracted from these records will enable linking Dutch and Asian personnel data for the VOC, providing insights into historical career trajectories. It utilizes the [Detectron2 framework](https://github.com/facebookresearch/detectron2) with a pretrained Mask R-CNN model (publay_mask_rcnn_R_50_FPN_3x). The training goals is to process tables to identify rows and columns, which we can then use for data extraction.


### Applications

- Historical research on VOC personnel records.
- Linking Dutch and Asian administrative records for comprehensive career tracking.
- Facilitating large-scale data extraction from historical handwritten archives.


### Repository Structure

- **data/**: Sample scanned images and annotations (structured in COCO format).
- **utils/**: Utility scripts for data preprocessing and annotation conversion.

