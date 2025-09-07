# Strategies for Crisis-Responsive Governance: Automated Anomaly Identification in Public Services

[![Paper-Link]](https://ojs.iscram.org/index.php/Proceedings/article/view/106) This repository contains the official implementation for the paper **"Strategies for Crisis-Responsive Governance: Automated Anomaly Identification in Public Services."**

Our work introduces a machine learning tool designed to help public service systems (like 311) automatically classify service requests and detect anomalies, which is especially critical during a crisis.

---

## 📜 Overview

During emergencies, public service systems are often overwhelmed with a surge in requests. Manually categorizing these requests and identifying emerging issues is slow and labor-intensive. This project provides a tool that:
1.  **Automatically classifies** service calls into predefined categories using a Support Vector Machine (SVM) model.
2.  **Detects anomalies** and irregular requests that may signal a new or growing crisis, enabling a faster response.

We validated this approach using data from the Orange County, Florida 311 System, with a specific focus on the COVID-19 period. The code in this repository allows for the full replication of our methodology.


## 🚀 Getting Started

Follow these instructions to set up the environment and run the code.

### 1. Prerequisites
Ensure you have Python 3.8+ installed. All required libraries are listed in the `requirements.txt` file.

### 2. Installation
Clone the repository and install the dependencies:

Clone this repository
git clone ....
cd Public-Service-Anomaly-Detection

Install the required libraries
pip install -r requirements.txt
 
### 3. Reproducing the Results
Follow these steps to run the full analysis pipeline as described in our paper.

Data Preparation
The Orange County 311 dataset used in our study is private and cannot be shared publicly. However, you can adapt this code for your own dataset.

To use the script, your data must be in CSV format and placed in the same directory as the script. The script expects the following files and column names:

311DataDump_Oct2020_Dec2021.csv

311DataDump_10292020_1403.csv

Both files must contain these three columns:

created_on: The timestamp of the service request.

category: The predefined service category (label).

issue_desc: The text description of the service request.

Run the Analysis
To reproduce the results, execute the Full_Code.py file. This single script runs the entire pipeline from data preprocessing to model evaluation and anomaly detection.

## How to Cite
If you use this code or our research in your work, please cite our paper:

```bibtex
@article{Unveren_Lehyeh_Pamukcu_Zobel_2024,
  title   = {Strategies for Crisis-Responsive Governance: Automated Anomaly Identification in Public Services},
  author  = {Unveren, Hakan and Lehyeh, Ayesh Abu and Pamukcu, Duygu and Zobel, Christopher W.},
  journal = {Proceedings of the International ISCRAM Conference},
  year    = {2024},
  month   = {May},
  doi     = {10.59297/evk7eh36},
  url     = {(https://ojs.iscram.org/index.php/Proceedings/article/view/106)}
}
