# Installation with Pixi

## Prerequisites
Install Pixi following the [official guide](https://pixi.sh/latest/#installation).

## Setup
```bash
git clone git@github.com:jiaming-zhou/X-ICM.git
cd X-ICM
pixi shell
```

Inside the Pixi shell, you can run tasks like:
- `pixi run setup_env` to install dependencies and download data/models.
- `pixi run eval_xicm "0,25,50,75,99" 25 Qwen2.5.7B.instruct 1 0,1,2,3,4,5,6,7 "lang_vis.out"` to evaluate the model.

For more tasks, run `pixi run --list`.
pip install -e .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

## 3. RLBench

Install the RLBench package:
```bash
cd RLBench
pip install -r requirements.txt
pip install -e .
```

## 4. YARR

Install the YARR package:
```bash
cd YARR
pip install -r requirements.txt
pip install -e .
```

## 5. X-ICM

Finally, install the dependencies for X-ICM:
```bash
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

## 6. [Optional] Setup Virtual Display

This is only required if you are running on a remote server without a physical display.

We provide a script to set up the virtual display in Ubuntu 20.04.

```bash
sudo apt update
sudo apt-get reinstall xorg freeglut3-dev libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb
sudo dpkg -i virtualgl*.deb
rm virtualgl*.deb
nohup sudo X &
```

Any later command using display (e.g., dataset generation and evaluation) should be run with `DISPLAY=:0.0 python ...`.

For more details, please refer to the [PyRep](https://github.com/stepjam/PyRep?tab=readme-ov-file#running-headless).

