### Introduction
The [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) API I built for the Roblox interview take-home assignment in 2022.
The API is wrriten in `script.py`, and the reasoning is written in `notes.md`

**The main purpose of this repo is to demonstrate my code and reasoning. The API itself could be outdated.**

### Requirement
Linux Ubuntu 16.04.3 LTS or higher

### Installation
1. Open up the terminal window in Linux (e.g., press Ctrl+Alt+T). You can refer to this instruction about how to open the terminal: https://www.howtogeek.com/686955/how-to-launch-a-terminal-window-on-ubuntu-linux/

2. Update the "apt-get" app with following command
`$ sudo apt-get update`

3. Install "Python 3.9" with following command
`$ sudo apt-get install python3.9`
`$ python3.9 --version`  (check the version of python3.9)

4. Update the "pip" app for "Python 3.9" with following command
`$ python3.9 -m pip install --upgrade pip`

5. Install the CPU version of "PyTorch" and "torchvision" with following command
`$ python3.9 -m pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html`

6. Install "OpenCV" with following command
`$ python3.9 -m pip install opencv-python`

7. Go to the project directory that contain the "script.py"
`$ cd /home/fuchen/Roblox_HW_2022`  (example)

### Usage
`$ python3.9 script.py [path]`
Run the face detection script on [path] and show the result. [path] could be an image file path or directory path.

`$ python3.9 script.py [path] --metric`
Same above, but also calculate and show the metrics.

`$ python3.9 script.py -h`
Show the description and usage of this script.

### Example
#### Run the script on the provided validation images
`$ python3.9 script.py val/`

#### Got the following results
```
1_faces_2.png   1
3_faces_3.jpg   3
0_faces_4.jpg   0
9_faces_1.jpg   8
1_faces_3.jpg   1
5_faces_2.jpg   5
1_faces_8.jpg   1
5_faces_3.jpg   7
4_faces_2.jpg   4
1_faces_6.jpg   1
2_faces_8.jpg   2
0_faces_7.jpg   0
5_faces_4.jpg   5
2_faces_7.jpg   2
1_faces_9.jpg   1
1_faces_5.jpg   1
4_faces_5.jpg   4
4_faces_3.jpg   2
1_faces_1.jpg   1
0_faces_5.jpg   0
0_faces_6.jpg   0
24_faces_1.jpg  24
7_faces_1.jpg   7
2_faces_6.jpg   2
12_faces_1.jpg  12
2_faces_3.jpg   3
2_faces_4.jpg   2
9_faces_2.jpg   9
1_faces_4.jpg   1
2_faces_2.jpg   2
1_faces_7.jpg   1
3_faces_2.jpg   4
2_faces_5.jpg   2
6_faces_1.jpg   6
precision: 0.968
recall: 0.976
f1_score: 0.972
False positives (wrong detections) per image: 0.1
False negatives (miss detections) per image: 0.1
```
