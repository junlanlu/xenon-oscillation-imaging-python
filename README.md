# xenon_ventilation

Ventilation imaging pipeline, developed at the [Driehuyslab](https://sites.duke.edu/driehuyslab/), processes MRI dicom data and produce summary report to analyze the functionality of the human lung. This README presents the installation procedure of the pipeline. Before moving to the installation process, download or clone this repository in your computer.

## Table of contents:

1. [Setup](#setup)

2. [Installation](#installation)

3. [Usage](#Usage)

4. [Acknowledgments](#acknowledgements)


## Setup

This pipeline is a cross platform program that works on Windows, Mac and Linux system. At least 8GB of RAM is required to run this pipeline. Windows users can install Windows Subsystem for Linux (WSL) or install Ubuntu as dual boot/in the virtual box. The details of WSL installation can be seen in Section 1.1. Warning: run time in WSL can be slower compare to Linux or Mac system.

Mac and Linux users skip to installation.  Note: Currently, this pipeline works on intel based Mac. Apple silicon based mac is not supported at this moment.

### 1.1. Windows Subsystem for Linux
Windows Subsystem for Linux installation process can seem overwhelming, especially following the procedure in the Microsoft [documentation](https://docs.microsoft.com/en-us/windows/wsl/install-win10). However a short YouTube video can make the install process much easier. One good example YouTube instruction can be seen [here](https://www.youtube.com/watch?v=X-DHaQLrBi8&t=385s&ab_channel=ProgrammingKnowledge2ProgrammingKnowledge2). Note: If the YouTube link is broken, please search in YouTube.

Next for opening any GUIs or graphical applications in WSL, you need to [install XMing](https://sourceforge.net/projects/xming/). Open the XMing when you want to run this pipeline. Write the following command in the terminal to show the GUI:

```
export DISPLAY=:0;
```
To avoide writing above command everytime, add this line to the `bashrc`. To find out the bashrc file, write `cd ~`, and open the file using a text editor. A text editor called `nano` can be installed using: sudo apt install nano

Now you can open the bashrc file using: nano bashrc ; then insert the command in the file and save it.

## Installation
### 2.1. Python Installation
First step of the installation process is to install python. This pipeline works with Python 3.9.1 in its current version. In order to install necessary Python Libraries, Python 3.9.1 version is required. To create a virtual environment, a 'conda' distribution is required. If you don't have conda distribution installed into your machine, you can install one downloading 'Anaconda' or 'Miniconda'. You can download the 'Anaconda Distribution' from this [link](https://www.anaconda.com/products/individual), or 'Miniconda' from this [link](https://docs.conda.io/en/latest/miniconda.html). Here, command line installation procedure has been presented. So, Mac user can download the Command Line Installer.

**Note**: if you have conda already installed, skip these steps

#### 2.1.1. Conda Installation on Mac or Linux:
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

If you don't see 'conda' directory after verifing, you can review the details of [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installation.

#### 2.1.2. Conda Installation on Windows Subsystem for Linux(WSL):
WSL users need to install `Anaconda` or `Miniconda` for Linux inside the WSL shell. Change your current directory to where you have downloaded your Anaconda or Miniconda installation file (.sh file).  Then run the:
```bash
bash filename
```
You can verify if you have `conda` now by typing `which conda` in your terminal.



### 2.2. Virtual Environment Creation and Python Package Installation

**Note**: if you have these packages already installed or a virtual environment already prepared, skip these steps

#### 2.2.1. Create Virtual Environment  
To create a virtual environment using `conda` execute the command in the terminal:

```bash
conda create --name xevent python=3.9.1
```
Here, `xevent` is the the given name, but any name can be given. 

To activate the environment, execute the command

```bash
conda activate xevent
```

#### 2.2.2. Install Required Packages
##### Installing Packages in the Virtual Environment:
Now we are ready to install necessary packages. Packages must be installed inside the virtual conda environment. The list of packages are in the `requirements.txt`. These two files can be found in the setup folder of the main program directory. If in the terminal you are not in the main program directory, change the directory using cd command. To install the required packages, execute the command:

```bash
pip install -r setup/requirements.txt
```

**Note:** if the above does not work, install the packages one-by-one starting from Tensorflow (most important).

**Note**: Junlan recommends installing packages one by one in the order they appear on the `requirements.txt` file

To confirm that correct packages are installed, execute the command

```
pip list
```
and verify that the packages in the virtual environment agree with that in the `requirements.txt` file. 



**Note for apple silicon users:** If you would like GPU support, follow this [guide](https://towardsdatascience.com/installing-tensorflow-and-jupyter-notebook-on-apple-silicon-macs-d30b14c74a08) for installing Tensorflow.



##### Install Packages in your Native Computer:

For Linux and WSL: 
```
sudo apt install wkhtmltopdf
sudo apt install poppler-utils
```
For Mac: 
```
brew install wkhtmltopdf
brew install poppler
```
### 2.3. Compilation and Download Necessary tools
#### 2.3.1 For Segmentation: Downloading the h5 models for machine learning
Check the `models/weights` folder if the `.h5` files are not available there, download  them from this [link](https://drive.google.com/drive/folders/1gcwT14_6Tl_2zkLZ_MHsm-pAYHXWtVOA?usp=sharing) and place it in the `models/weights` folder in your main program directory.

#### 2.3.3. For Registration: Compiling ANTs
If you already have the `N4BiasfieldCorrection` , `antsRegistration` and `antsApplyTransforms` executables, skip this step. 

Compiling ANTs require to install git, cmake, g++, zlib. Following commands will install these packages.

#### Linux and Windows Subsystem For Linux(WSL) user: execute following commands on your terminal:

```bash
sudo apt-get -y install git
sudo apt-get -y install cmake
sudo apt install g++
sudo apt-get -y install zlib1g-dev
```

#### Mac User: check you have git, cmake, g++ writing, e.g. which git, which cmake
If you don't have any of these, you have to install in the command line.Now you can install packages writing following commands:  
```bash
brew install git
brew install cmake
brew install g++
```
Now we are ready to perform SuperBuild. Execute the following commands on your terminal.
```bash
workingDir=${PWD}
git clone https://github.com/ANTsX/ANTs.git
mkdir build install
cd build
cmake \
    -DCMAKE_INSTALL_PREFIX=${workingDir}/install \
    ../ANTs 2>&1 | tee cmake.log
make -j 4 2>&1 | tee build.log
cd ANTS-build
make install 2>&1 | tee install.log
```
Warning: This might take a while. 

After successful ANTs SuperBuild, you will have to copy `antsApplyTransforms`, `antsRegistration` and `N4BiasFieldCorrection` files from the `install/bin`, and paste it to the `bin`  directory. Now we are ready to process MRI scan of human lung.

Note: If neccesary, the details of ANTs Compilation can be seen [here](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS)

## Usage

### 3.1. Team Xenon Workflow for External Sites (Such as UVA, UM, SickKids, Western, etc.)

 ```bash
   conda activate xevent #activates the conda environment
   python run_gui.py
 ```
The above comands should open the GUI. 

WSL user: Did you write "export DISPLAY=:0;" in your terminal? Is your Xming open? If not, open the Xming and write "export DISPLAY=:0;" in the WSL terminal. Then execute the above commands.

### 3.2. Team Xenon Workflow for Duke Data Processing

Warning: this is the Team Xenon workflow only. Other users do not have to follow the exact procedures.

1. Create a new subject folder. This will typically have the format of `###-###` or `###-###X`.
2. Then log onto the `smb://duhsnas-pri/xenon_MRI_raw/` drive and enter the directory of interest corresponding to the recently scanned subject. Copy the files on your computer. Determine how many dedicated ventilation scans are there (usually 1 or 2). If there is only 1, create a subfolder named `###-###` in your new subject folder and copy all twix files and dicom files into that subfolder. If there are 2 sets of scans, create subfolders `###-###_s1` (for the first scan) and `###-###_s2`(for the second scan) and copy the twix/dicom files corresponding to the first scan (ventilation and proton) and copy the twix files corresponding the second set of scans into the other.


3. Navigate to the repository folder (xenon_ventilation) using the command line. Before you run any code, make sure you have the latest updates and are processing on the master branch. You can do this by

   ```shell
   git pull
   git checkout master
   ```

4. Activate the conda environment

   ```shell
   conda activate xevent #activates the conda environment
   ```

5. **Run using the command line:** Create a config file using a text-editor, and copy `base_config.py` format. Rename the file to the patient name, being able to identify it later on and also safe it into the `config` folder. Edit the fields to configure the pipeline. Then run

   ```shell
   python main_mapping.py --config config/[yourconfigfile].py
   ```

   **Run using the GUI**: Launch the GUI using the command below and edit the widgets in the GUI to configure the pipeline

   ```shell
   python run_gui.py
   ```

   

   1. When the program is done running, it will automatically create a folder `Dedicated_Ventilation` folder with all the analyzed files inside
   2. Edit the segmentation as needed (if not, you are done)
      1. Run the pipeline again, selecting the appropriate segmentation flag and pointing to the segmentation file
   3. Copy all the contents in the subject folder and paste it into `smb://duhsnas-pri/duhs_radiology/Private/TeamXenon/01_ClinicalOutput/Processed_Subjects`
   4. Upload reports to Slack


## Acknowledgements:

Original Author: Mu He

Past and current Developers: Ziyi Wang, David Mummy, Junlan Lu, Suphachart Leewiwatwong, Isabelle Dummer, and Sakib Kabir.

Correspondence: Junlan Lu (junlan.lu@duke.edu) ; David Mummy (david.mummy@duke.edu)
