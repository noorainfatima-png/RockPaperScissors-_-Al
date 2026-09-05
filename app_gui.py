import os
import csv
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk

MODEL_PATH = "models/rps_model.pkl"
DATA_PATH = "data/moves.csv"

# Load initial model
model = joblib.load(MODEL_PATH)

move_map = {"rock": 0, "paper": 1, "scissors": 2}
reverse_map = {0: "rock", 1: "paper", 2: "scissors"}
counter_move = {"rock": "paper", "paper": "scissors", "scissors": "rock"}

history = []
user_wins = 0
ai_wins = 0
ties = 0
total_games = 0
correct_predictions = 0

def retrain_model_on_the_fly():
    """Retrains the model safely using data/moves.csv."""
    global model
    try:
        # Read CSV with on_bad_lines='skip' to avoid crash on mismatched rows
        data = pd.read_csv(DATA_PATH, on_bad_lines='skip')
        data.columns = data.columns.str.strip()

        cols = ["Move1", "Move2", "Move3", "NextMove"]
        for c in cols:
            if c in data.columns:
                data[c] = data[c].astype(str).str.strip().str.lower().map(move_map)

        data = data.dropna()

        X = data[["Move1", "Move2", "Move3"]]
        y = data["NextMove"]

        new_model = RandomForestClassifier(
            n_estimators=150, 
            max_depth=12, 
            class_weight="balanced", 
            random_state=42
        )
        new_model.fit(X, y)
        model = new_model
        joblib.dump(model, MODEL_PATH)
        print("Model retrained successfully on live data!")
    except Exception as e:
        print("Retraining error:", e)

def record_move_to_csv(prev_3_moves, current_move):
    """Appends new move sequence cleanly to CSV."""
    file_exists = os.path.exists(DATA_PATH)
    with open(DATA_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Move1", "Move2", "Move3", "NextMove"])
        writer.writerow(list(prev_3_moves) + [current_move])

def play(user_choice):
    global user_wins, ai_wins, ties, total_games, correct_predictions, history

    history.append(user_choice)

    # Need at least 3 previous moves to start predicting
    if len(history) < 4:
        rounds_left = 4 - len(history)
        lbl_status.config(
            text=f"Warm-up ({len(history)}/3). Need {rounds_left} more move(s) to activate AI."
        )
        return
    else:
        lbl_status.config(text="🔥 Adaptive AI Active (Learning Your Play Style Live!)")

    # Extract 3 previous moves
    prev_moves = history[-4:-1]

    # Save interaction to CSV for live learning
    record_move_to_csv(prev_moves, user_choice)

    # Format input for prediction
    input_df = pd.DataFrame(
        [[move_map[prev_moves[0]], move_map[prev_moves[1]], move_map[prev_moves[2]]]],
        columns=["Move1", "Move2", "Move3"]
    )

    predicted_user_move = reverse_map[model.predict(input_df)[0]]
    ai_choice = counter_move[predicted_user_move]

    total_games += 1

    if predicted_user_move == user_choice:
        correct_predictions += 1

    # Determine round result
    if user_choice == ai_choice:
        result = "Result: Tie! 🤝"
        ties += 1
    elif counter_move[user_choice] == ai_choice:
        result = "Result: AI wins this round! 🤖"
        ai_wins += 1
    else:
        result = "Result: You win this round! 🎉"
        user_wins += 1

    # Retrain model every 10 games
    if total_games % 10 == 0:
        retrain_model_on_the_fly()

    # Calculate statistics
    ai_accuracy = (correct_predictions / total_games) * 100
    win_rate = (user_wins / total_games) * 100

    # Update Labels
    lbl_prediction.config(text=f"AI Predicted: {predicted_user_move.upper()}")
    lbl_ai_choice.config(text=f"AI Played: {ai_choice.upper()}")
    lbl_result.config(text=result)

    lbl_score.config(text=f"You: {user_wins}  |  AI: {ai_wins}  |  Ties: {ties}")
    lbl_stats.config(
        text=f"Total Games: {total_games}  |  Win Rate: {win_rate:.1f}%\nAI Prediction Accuracy: {ai_accuracy:.1f}%"
    )

# GUI Setup
root = tk.Tk()
root.title("Adaptive Rock Paper Scissors AI")
root.geometry("440x500")

lbl_title = tk.Label(root, text="Adaptive Rock Paper Scissors AI", font=("Arial", 15, "bold"))
lbl_title.pack(pady=10)

lbl_status = tk.Label(root, text="Click a button to start!", font=("Arial", 10))
lbl_status.pack(pady=2)

lbl_prediction = tk.Label(root, text="", font=("Arial", 11))
lbl_prediction.pack(pady=2)

lbl_ai_choice = tk.Label(root, text="", font=("Arial", 11))
lbl_ai_choice.pack(pady=2)

lbl_result = tk.Label(root, text="", font=("Arial", 12, "bold"))
lbl_result.pack(pady=8)

frame_stats = tk.LabelFrame(root, text=" Performance Stats ", font=("Arial", 10, "bold"), padx=10, pady=10)
frame_stats.pack(pady=10, fill="both", expand=True, padx=20)

lbl_score = tk.Label(frame_stats, text="You: 0  |  AI: 0  |  Ties: 0", font=("Arial", 11, "bold"))
lbl_score.pack(pady=4)

lbl_stats = tk.Label(
    frame_stats, 
    text="Total Games: 0  |  Win Rate: 0.0%\nAI Prediction Accuracy: 0.0%", 
    font=("Arial", 10)
)
lbl_stats.pack(pady=4)

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=15)

tk.Button(frame_buttons, text="🪨Rock", font=("Arial", 11), width=10, command=lambda: play("rock")).grid(row=0, column=0, padx=4)
tk.Button(frame_buttons, text="📄Paper", font=("Arial", 11), width=10, command=lambda: play("paper")).grid(row=0, column=1, padx=4)
tk.Button(frame_buttons, text="✂️Scissors", font=("Arial", 11), width=10, command=lambda: play("scissors")).grid(row=0, column=2, padx=4)

root.mainloop()