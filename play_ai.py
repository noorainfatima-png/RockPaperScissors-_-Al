import joblib
import pandas as pd

model = joblib.load("models/rps_model.pkl")

move_map = {"rock": 0, "paper": 1, "scissors": 2}
reverse_map = {0: "rock", 1: "paper", 2: "scissors"}
counter_move = {"rock": "paper", "paper": "scissors", "scissors": "rock"}

history = []
user_score = 0
ai_score = 0

print("=== ROCK PAPER SCISSORS AI GAME ===")
print("Play against the AI! Type 'quit' anytime to exit.\n")

# Collect 3 initial moves to start pattern tracking
while len(history) < 3:
    move = input(f"Enter warm-up move {len(history)+1} (rock/paper/scissors): ").strip().lower()
    if move in move_map:
        history.append(move)
    else:
        print("Invalid move. Enter rock, paper, or scissors.")

print("\n--- Warm-up complete! Game starting ---")

while True:
    input_df = pd.DataFrame(
        [[move_map[history[-3]], move_map[history[-2]], move_map[history[-1]]]],
        columns=["Move1", "Move2", "Move3"]
    )
    
    # AI predicts your move and selects the counter move
    predicted_user_move = reverse_map[model.predict(input_df)[0]]
    ai_choice = counter_move[predicted_user_move]

    user_move = input("\nYour move (rock/paper/scissors): ").strip().lower()
    if user_move == "quit":
        break
    if user_move not in move_map:
        print("Invalid move! Try again.")
        continue

    history.append(user_move)

    print(f"AI predicted you would play: {predicted_user_move}")
    print(f"AI played: {ai_choice}")

    if user_move == ai_choice:
        print("Result: Tie!")
    elif counter_move[user_move] == ai_choice:
        print("Result: AI wins this round! 🤖")
        ai_score += 1
    else:
        print("Result: You win this round! 🎉")
        user_score += 1

    print(f"Scoreboard -> You: {user_score} | AI: {ai_score}")

print(f"\nFinal Score -> You: {user_score} | AI: {ai_score}")