# Dataset

## Dataset

This project is designed for use with the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset.

ADNI provides longitudinal MRI scans, cognitive assessments, and clinical metadata collected from multiple medical centers.

Official Website

https://adni.loni.usc.edu/

---

## Dataset Structure

Expected folder structure:

data/

    raw/

        CN/

        MCI/

        AD/

    processed/

---

## MRI Preprocessing

The preprocessing pipeline consists of:

1. DICOM to NIfTI conversion
2. MRI intensity normalization
3. Slice extraction
4. Image resizing (224×224)
5. Data augmentation
6. Dataset splitting

---

## Clinical Features

Optional metadata may include:

- Age
- Gender
- Education
- MMSE
- CDR
- APOE genotype (if available)

---

## Data Augmentation

Training images may undergo

- Horizontal Flip
- Rotation
- Random Crop
- Brightness Adjustment
- Normalization

to improve model robustness.

---

## Dataset Availability

The ADNI dataset is not included in this repository due to licensing restrictions.

Researchers may request access through the official ADNI website.
