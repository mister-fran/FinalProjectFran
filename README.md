# Master Thesis: Angular Reconstruction of High Energy Neutrino Events in IceCube using Machine Learning

**Author**: Luc Voorend  
**Thesis Date**: May 20, 2025  
**Code base language**: Python 🐍
## 🧊 Overview

Welcome! This repository hosts the complete codebase for my Master's thesis. The core focus is developing and evaluating a **transformer-based machine learning model** for reconstructing the arrival direction (angular reconstruction) of high-energy neutrino events detected by the **IceCube Neutrino Observatory** at the South Pole.

---

### 🌌 Motivation: Unveiling the High-Energy Universe

Physics has made incredible strides with the **Standard Model** ⚛️, but the universe still holds many mysteries, especially where particle physics meets astrophysics. Understanding extreme cosmic events requires deciphering high-energy particle interactions. While we detect particles from powerful cosmic accelerators, their origins remain largely unknown.

Enter the **neutrino** 👻 – the elusive "*ghost particle*". With almost no mass and no charge, neutrinos travel billions of light-years unimpeded, carrying direct information from their sources. Unlike γ-rays or cosmic rays, they aren't easily absorbed or deflected. This makes them perfect messengers for multi-messenger astronomy. The **IceCube Observatory** was built precisely for this, aiming to map the high-energy neutrino sky and has already linked neutrinos to distant cosmic sources.

---

### 🎯 The Challenge: Pinpointing Neutrino Sources

Identifying specific neutrino sources requires *excellent angular resolution*. IceCube primarily uses muon neutrinos ($\mu_\nu$) for their track-like signatures.
* **Traditional Methods:** Likelihood-based techniques achieve good resolution (e.g., ≲ 1° at TeV) but are computationally *slow*.
* **Machine Learning:** GNNs showed promise for speed but haven't consistently beaten traditional methods at high energies (> GeV). The 2023 IceCube Kaggle competition put the spotlight on the potential of **transformers**.

---

### ✨ This Project: Transformers & PMT-fication

This thesis leverages these recent advancements, introducing a novel approach called **PMT-fication** (aggregating detector pulses at the Photomultiplier Tube level) to optimize data for a **transformer model**. The code for PMT-fication is released as a separate toolkit [IcePACK](https://github.com/KUcyans/IcePack/tree/main) The key goals of the thesis were:

* 🧠 Develop a transformer capable of reconstructing muon neutrino tracks (100 GeV - 100 PeV).
* 🔬 Investigate how factors like *event selection*, *model size*, *training data size*, and *input representation* (PMT-fication) affect performance.
* 📈 Compare the final model's angular resolution against the state-of-the-art **SplineMPE** likelihood method.

Ultimately, this repository aims to share these findings and provide a **reproducible and extendable** codebase for future research in neutrino astronomy.

---

## 📄 Thesis Document

The full thesis document, containing detailed theoretical background, methodology, analysis, results, and discussion, is available in the root directory of this repository:

* [Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_LucVoorend.pdf](https://github.com/lucvoorend/IceCubeTransformer/blob/master/Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_LucVoorend.pdf) 

Part I covers the theoretical background (Standard Model, Neutrino Physics, IceCube, Machine Learning, Transformers, Traditional Reconstruction). Part II details the specific methods, data, model architecture, training, results, and conclusions of this research.

## 📁 Repository Structure

This codebase is organized into three main directories, plus the thesis PDF and requirements:

├── **Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_LucVoorend.pdf** # The full thesis document  
├── data_preparation/ # Scripts for data cleaning and preparation  
│ └── CR_cleaning.py   
├── training_and_inference/ # Core scripts for the transformer model  
│ ├── src/  
│ │ ├── model.py # Transformer model class definition  
│ │ ├── dataset.py # Custom Dataset class for IceCube data  
│ │ ├── dataloader.py # Dataloader implementation (using PMT-fication)  
│ │ ├── loss.py # Loss function(s) used for training  
│ │ ├── utils.py # Assertion functions for config file  
│ ├── train.py # Script to train the model  
│ ├── inference.py # Script to run inference and evaluate the model  
│ └── config.yaml # Config file controlling settings for training and inference  
├── analysis/ # Analysis notebook
│ └── analysis.ipynb # Jupyter notebook to generate figures from the thesis  
├── requirements.txt # Python dependencies  
└── README.md # This file  

* **`data_preparation/`**: Contains all scripts related to selecting cosmic ray cleaned events from i3 files.
* **`training_and_inference/`**: This is the core directory. It holds the Python code defining the transformer architecture, the custom dataset and dataloader logic handling the PMT-fied data, loss functions, and the main scripts for training the model (`train.py`) and performing angular reconstruction on new data (`inference.py`).
* **`analysis/`**: Includes Jupyter notebooks (`.ipynb`) to generate the plots, figures, and statistical analyses presented in the thesis.

## 🚀 Getting Started

Follow these steps to set up the environment and run the code.

### Prerequisites

* Python (version 3.9+)
* Required Python packages (see `requirements.txt`)
* Access to the relevant IceCube dataset(s) (See Thesis Part II, Section 6.1 for details on data) 

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1.  **Data Preparation:**
    * If not yet processed, apply the [IcePACK](https://github.com/KUcyans/IcePack/tree/main) toolkit for PMT-fication of the input data.
    * Follow the `readme` of the directory for specific instructions.

2.  **Training:**
    * Configure the training parameters in `config.yaml`.
    * Run the training script `train.py`
    ```bash
    # For no hang up run on a Unix/Linux system, use:
    nohup python -u training_and_inference/train.py > logs/output.log &
    ```
3.  **Inference & Evaluation:**
    * Use the trained model to perform angular reconstruction on a test dataset.
    * Add trained model details to `config.yaml` and select data files for inference
    ```bash
    python training_and_inference/inference.py
    ```
4.  **Analysis:**
    * Open and run the Jupyter notebook(s) in the `analysis/` directory to reproduce the plots and figures from the thesis.
    ```bash
    jupyter notebook analysis/analysis.ipynb
    ```

*Note: Specific command-line arguments and configurations might vary. Please refer to the scripts and the thesis document for detailed usage.*

## 🔄 Reproducibility

This repository aims to ensure the reproducibility of the results presented in the thesis.
* The `requirements.txt` file lists the necessary package versions.
* The scripts in `data_preparation/` allow for recreating the exact data processing steps, combined with the IcePACK library.
* The `training_and_inference/` scripts, along with the training configurations, enable retraining or re-evaluating the model.
* The `analysis/analysis_plots.ipynb` notebook uses the output from the inference step to generate the key figures, allowing for direct comparison with the thesis results.

Please consult Part II of the thesis for details on the specific datasets, simulation parameters, event selection criteria, and evaluation metrics used.

## ✨ Extending the Work

This research opens several avenues for future exploration:
* Investigating different transformer architectures or attention mechanisms.
* Applying the PMT-fication and transformer approach to other reconstruction tasks (e.g., energy reconstruction, particle identification).
* Exploring alternative pulse aggregation or data representation techniques.
* Training on larger or more diverse datasets.
* Extending from Monte Carlo to IceCube real data events.

Feel free to fork this repository and build upon this work. Contributions and suggestions are welcome!

## 🙏 Acknowledgements

* The code related to PMT-fication and event selection was largely written by [Cyan Jo](https://github.com/KUcyans).
* The CR_cleaning code has been written by [Johann Nikolaides](https://github.com/jn1707).
* Special thanks to [Troels Petersen](https://github.com/troelspetersen), [Inar Timiryasov](https://github.com/timinar) and [Jean-Loup Tastet](https://github.com/JLTastet) for supervising the project.

## 📜 Citation

If you use this code or build upon the methods presented in this thesis, please cite the work appropriately.

```bibtex
@mastersthesis{Voorend_2025,
  author       = {Luc Voorend},
  title        = {Angular reconstruction of high energy neutrinos using machine learning},
  school       = {University of Copenhagen},
  year         = {2025},
  url          = {https://github.com/lucvoorend/IceCubeTransformer}
}
```

## 📝 License

This repository contains different types of content under different licenses:

### Code

All source code files (typically files with extensions like `.py` and `.yaml`) are licensed under the **MIT License**. You can find the full license text in the [LICENSE](LICENSE) file. This means you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code, provided the original copyright notice and license text are included.

### Thesis Document

The PDF document located at [Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_LucVoorend.pdf](https://github.com/lucvoorend/IceCubeTransformer/blob/master/Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_LucVoorend.pdf) contains the author's thesis.  
**Copyright © 2025 L.H. Voorend.  
All Rights Reserved.**

This document is provided for viewing and academic citation purposes only.
* **You MAY:** View the document, cite the document according to academic standards.
* **You MAY NOT:** Modify the document, distribute copies of the document, host the document elsewhere, or use it for commercial purposes without explicit written permission from the author.
