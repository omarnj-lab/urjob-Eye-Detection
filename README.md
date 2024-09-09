# Video Eye Contact Analysis

This Python script analyzes eye contact frequency in video files using OpenCV and Dlib.

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/omarnj-lab/urjob-Eye-Detection.git
cd urjob-Eye-Detection
pip install -r requirements.txt
```

## Usage
Run the script by specifying the path to your video and how many frames to skip:

```bash
python eye_contact_analyzer.py path_to_video.mp4 10
```

## Output

The script outputs a JSON file eye_contact_results.json that includes detailed analytics about the eye contact in the video.
