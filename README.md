# xenon-oscillation-imaging python

The oscillation imaging pipeline, developed at the [Driehuyslab](https://sites.duke.edu/driehuyslab/). Before moving to the installation process, download or clone this repository.

## Table of contents:

1. [Setup](#setup)

2. [Installation](#installation)

3. [Usage](#Usage)

4. [Acknowledgments](#acknowledgements)


## Setup

This pipeline is a cross-platform program that works on Windows, Mac, and Linux systems. At least 8GB of RAM is required to run this pipeline. Windows users can install Windows Subsystem for Linux (WSL) or install Ubuntu as dual boot/in the virtual box. The details of the WSL installation can be seen in Section 1.1. Warning: run time in WSL can be slower than Linux or Mac systems.

Mac and Linux users skip to installation.  Note: Currently, this pipeline works on Intel-based Mac. Apple silicon-based mac is not supported at this moment.

### 1.1. Windows Subsystem for Linux
Windows Subsystem for Linux installation process can seem overwhelming, especially following the procedure in the Microsoft [documentation](https://docs.microsoft.com/en-us/windows/wsl/install-win10). However, a short YouTube video can make the installation process much easier. One good example of YouTube instruction can be seen [here](https://www.youtube.com/watch?v=X-DHaQLrBi8&t=385s&ab_channel=ProgrammingKnowledge2ProgrammingKnowledge2). Note: If the YouTube link is broken, please search on YouTube.

## Installation
### 2.1. Python Installation
The first step of the installation process is to install python. This pipeline works with Python 3.9.1 in its current version. To install the necessary Python Libraries, Python 3.9.1 version is required. To create a virtual environment, a 'conda' distribution is required. If you don't have conda distribution installed on your machine, you can install one by downloading 'Anaconda' or 'Miniconda'. You can download the 'Anaconda Distribution' from this [link](https://www.anaconda.com/products/individual), or 'Miniconda' from this [link](https://docs.conda.io/en/latest/miniconda.html). Here, the command-line installation procedure has been presented. So, Mac users can download the Command Line Installer.

**Note**: if you have conda already installed, skip these steps

#### 2.1.1. Conda Installation on Intel Mac or Linux:
Now, open the terminal. You need to write a command to install Anaconda/Miniconda. Command to install the Anaconda or Miniconda is:

```bash
bash ~/path/filename
```
Example: If your downloaded Anaconda file is in the "Downloads" folder and the file name is "Anaconda3-2020.11-Linux-x86_64.sh", write the following in the terminal:

```bash
bash ~/Downloads/Anaconda3-2020.11-Linux-x86_64.sh
```
Here, path = Downloads and filename = Anaconda3-2020.11-Linux-x86_64.sh

Press "enter" and reply "yes" to agree to the license agreements. After completing the installation process, close and reopen the terminal. You can verify if you have `conda` now by typing `which conda` in your terminal.

If you don't see 'conda' directory after verifying, you can review the details of [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installation.

#### 2.1.2. Conda Installation on Windows Subsystem for Linux (WSL):
WSL users must install `Anaconda` or `Miniconda` for Linux inside the WSL shell. Change your current directory to where you have downloaded your Anaconda or Miniconda installation file (.sh file).  Then run the:
```bash
bash filename
```
You can verify if you have `conda` now by typing `which conda` in your terminal.

#### 2.1.3. Conda Installation on Apple Silicon Mac:

Follow step 2 in this [guide](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706) to install Miniforge. This is required as a conda installer for some reason so that you can properly install tensorflow plugins.

### 2.2. Virtual Environment Creation and Python Package Installation

**Note**: if you have these packages already installed or a virtual environment already prepared, skip these steps

#### 2.2.1. Create Virtual Environment  
To create a virtual environment using `conda` execute the command in the terminal:

```bash
conda create --name xevent python=3.9.1
```
Here, `xevent` is the given name, but any name can be given. 

To activate the environment, execute the command

```bash
conda activate xeosc
```

#### 2.2.2. Install Required Packages
##### Installing Packages in the Virtual Environment:
Now we are ready to install the necessary packages. Packages must be installed inside the virtual conda environment. The list of packages is in the `requirements.txt`. These two files can be found in the setup folder of the main program directory. If in the terminal you are not in the main program directory, change the directory using cd command. To install the required packages, execute the command:

```bash
pip install -r setup/requirements.txt
```

**Note:** If the above does not work, delete and recreate the conda virtual environment and start over. If the above does not work, install the packages one-by-one starting from Tensorflow (most important).

**Note**: Junlan recommends installing packages one by one in the order they appear on the `requirements.txt` file

To confirm that the correct packages are installed, execute the command

```
pip list
```
and verify that the packages in the virtual environment agree with that in the `requirements.txt` file. 

**Note for apple silicon users:** If you would like GPU support, follow this [guide](https://towardsdatascience.com/installing-tensorflow-and-jupyter-notebook-on-apple-silicon-macs-d30b14c74a08) for installing Tensorflow.

### 2.3. Download Necessary tools
#### 2.3.1 For Segmentation: Downloading the h5 models for machine learning
Check the `models/weights` folder if the `.h5` files are not available there, download them from this [link](https://drive.google.com/drive/folders/1gcwT14_6Tl_2zkLZ_MHsm-pAYHXWtVOA?usp=sharing) and place them in the `models/weights` folder in your main program directory.

## Usage

1. Create a new subject folder inside the `data` directory. This folder should contain at the minimum dynamic spectroscopy and 1-point dixon imaging twix files.


3. Navigate to the repository folder (xenon_ventilation) using the command line. Before you run any code, make sure you have the latest updates and are processing on the master branch. You can do this by

   ```shell
   git pull
   git checkout master
   ```

4. Activate the conda environment

   ```shell
   conda activate xeosc #activates the conda environment
   ```

5. **Run using the command line:** Create a config file using a text editor, and copy  the`demo_config.py` format. Rename the file to the patient's name, being able to identify it later on, and also save it into the `config` folder. Edit the fields inside the config as necessary to configure the pipeline. Then run

   ```shell
   python main.py --config config/[your_config_file].py
   ```



## Acknowledgments:

Original Author: Junlan Lu

Credits: PJ Niedbalski, Elianna Bier, Sakib Kabir, Ziyi Wang

Correspondence: Junlan Lu (junlan.lu@duke.edu) ; Bastiaan Driehuys (bastiaan.driehuys@duke.edu)