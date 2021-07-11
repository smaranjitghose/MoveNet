# Efficient Pose Estimation

This repo comprises of a script that helps us to detect 17 keypoints from the human body and draw the corresponding edges for Pose Detection.

**MODEL USED** : MoveNet Lightning SinglePose

## Usage:

- Clone this repository

`git clone www.github.com/smaranjitghose/movenet`

- Move inside the cloned directory of the project

`cd movenet`

- Make sure Python is installed and set to PATH

- Install the dependencies [Creating a virtual Environment may be useful beforehand]

`pip install - requirements.txt`

- Run the script

`python main.py`

# Note:

- For Linux/Mac based systems, replace the `python` with `python3` in the above command(s)
- In order to get a live demonstration, open the main.py script in a text editor or IDE like VS Code and change the parameter passed to the `cv2.VideoCapture()` function to 0 or 1 to obtain the feed in real time from your camera
- In order to try this on other videos, repeat the above step and change the update the variable input_video_path.
- For ensuring efficacy in particular scenarios (which might be dependent on factors like lightning, skin color, background light and age), try tweaking the threashold value
- Remember, this is solely for prototyping purposes. For better results, use the Thunder Version.

# References/Credits:

[TFHUB MoveNet](https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3)
