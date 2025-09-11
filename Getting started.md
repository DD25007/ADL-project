## Getting Started

The `./data` folder is added to `.gitignore` to avoid overloading local storage and will not be uploaded to this repo. Please follow the below steps to download and use the datasets.

### Clone this git repo into your local machine

1. Make sure you have git installed in your system.

2. Open a terminal and clone the repository to your local machine:
    ```bash
    git clone https://github.com/DD25007/ADL-project.git
    cd ADL-project
    ```


### Install python packages
The following commands will create a python environment and downlaod the required packages.
1. `python3 -m venv .venv`

2. `pip install -r requirements.txt` 

### Steps to Download and Extract Kaggle Dataset

1. **Get Kaggle API Credentials**

    * Go to Kaggle Account Settings
    * Click "Create New API Token"
    * This downloads a file called kaggle.json

2. **Place kaggle.json in the correct location**
    * Move kaggle.json to the .kaggle directory in your home folder:
    ```bash
    mkdir -p ~/.kaggle
    mv /path/to/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

3. **Download and Unzip the Dataset**

    In your notebook or terminal, run:

    `!kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p ./data --unzip`

### Steps to Download and Use Dataset from Hugging Face

1. **Get Hugging Face API Credentials**
    * Authenticate with Hugging Face `hf auth login`
    * Follow the prompt to paste your Hugging Face token [get it here](https://huggingface.co/settings/tokens).

2. **Download the Dataset in Your Script or Notebook**
    ```python
    from datasets import load_dataset

    # Download and cache the dataset in the specified folder
    ds = load_dataset("Amss007/ultrasound_dataset_v3_1", cache_dir="./data")
    print(ds)
    ```
