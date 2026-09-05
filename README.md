# RockPaperScissors-_-Al

An AI-powered desktop game built with Python! It studies your moves, spots your habits, and learns how to beat you in real time using Machine Learning. 

##  Features

* **Smart Prediction:** Tracks your last 3 moves to guess what you will play next.
* **Learns As You Play:** Automatically updates its brain every 10 rounds to counter your personal play style.
* **Live Stats Dashboard:** Displays your Win Rate, Total Games, and AI Accuracy in real time.
* **Easy Graphical Interface:** Clean Tkinter desktop window with simple clickable buttons.

## 📁 Project Structure

```text
RockPaperScissors_AI/
│── data/
│   └── moves.csv
│── models/
│   └── rps_model.pkl
│── train_model.py
│── test_model.py
│── play_ai.py
│── app_gui.py
│── requirements.txt
└── README.md

```

##  Requirements

* Python 
* `pandas`
* `scikit-learn`
* `joblib`

##  How to Run

1. **Install dependencies:**
```bash
pip install -r requirements.txt

```


2. **Train the AI:**
```bash
python train_model.py

```



3. **Launch the Game:**
```bash
python app_gui.py

```
