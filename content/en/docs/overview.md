---
title: Software Overview
description: The Idea Behind AmyPET - Robust Preprocessing For Solid Results
weight: 1
---

{{% pageinfo %}}

The aim of AmyPET software package is to provide open-source platform supporting high accuracy and precision PET image analysis in neurodegeneration and other neurological disease.  The emphasis is put on preparing the input images for robust analysis, which involves image frame alignment, including motion detection and correction, partial volume correction, etc.

{{% /pageinfo %}}


AmyPET is an advanced and open-source software package written in Python to provide all necessary tools for image analysis of static and dynamic PET image data.  The recommended image data is the raw DICOM image data exported from a PET scanner without modifications to obtain all necessary information about the frame timings, the PET radiopharmaceutical used, injection time, etc.

The key features include:

- Automated calibration with the original GAAIN PET and MR datasets for each tracer and for any modification of the original/base Centiloid pipeline.
- Dedicated for research purposes by enabling highly flexible processing pipelines (e.g., native or MNI PET space sampling, generates CL images in any space)
- Comprehensive Quality Control for each stage of processing – generates QC output folders for alignment, image registration, spatial non-rigid normalisation.
- Support for dynamic coffee break protocols and advanced alignment using PET frames or CT scans.
- Support for amyloid, tau and TSPO tracers.

## Multi-frame PET image alignment

![png](/AmyPET/figs/ovrv_1.png)

## Displacement Metric

![png](/AmyPET/figs/ovrv_2.png)