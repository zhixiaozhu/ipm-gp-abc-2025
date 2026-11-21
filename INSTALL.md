# Installation

These steps are for readers who have not used Python before.

## 1. Install Python 3.11

The code needs **Python 3.11**: TensorFlow 2.13 used here cannot be installed on Python 3.12 or later. Go to the [Python 3.11.9 page](https://www.python.org/downloads/release/python-3119/) on python.org, scroll down to **Files** at the bottom of the page, and download "macOS 64-bit universal2 installer" or "Windows installer (64-bit)". Run the installer with the default options.

## 2. Move into the repository folder

The repository folder is the `ipm-gp-abc-2025` folder you downloaded, which contains this file. If you downloaded a `.zip` file, unzip it first; the folder name may differ from `ipm-gp-abc-2025`. 

Open a terminal: on macOS, open the Terminal app; on Windows, open Command Prompt (press the Windows key, type `cmd` and press Enter). Then type `cd` followed by a space, drag the repository folder onto the terminal window, and press Enter. Alternatively, type the location of the folder yourself, for example `cd ~/Downloads/ipm-gp-abc-2025`.

All the commands in the following steps are run from the repository folder.

## 3. Create a virtual environment

A virtual environment is a folder, here called `.venv` and created inside the repository folder, that holds the packages for this project only, so they do not affect other Python projects (similar in spirit to `renv` in R). Create and activate it:

macOS:

```
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows:

```
py -3.11 -m venv .venv
.venv\Scripts\activate
```

The terminal prompt now starts with `(.venv)`.

## 4. Install the packages

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs the packages at the versions listed in `requirements.txt` (about 2 GB; this takes a few minutes). When it has finished, the last line starts with `Successfully installed`. If the installation fails while building `pytensor` (this can happen on Apple Silicon Macs), run `xcode-select --install` and repeat this step.

## 5. Run the vignette

```
jupyter lab toy_example/vignette.ipynb
```

JupyterLab opens in your web browser. Choose **Run > Run All Cells**. The vignette takes about three minutes to run. Keep the terminal open while you use JupyterLab; when you have finished, choose **File > Shut Down** in JupyterLab.

Next time, repeat Step 2, activate the environment (`source .venv/bin/activate` on macOS, `.venv\Scripts\activate` on Windows), then run Step 5.

