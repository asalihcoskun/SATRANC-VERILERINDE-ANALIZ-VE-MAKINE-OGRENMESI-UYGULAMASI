import numpy as np

def classify_game_phase(move_count, piece_count, pawn_count):
    if move_count <= 15:
        return "Opening"
    elif piece_count <= 14:
        return "Endgame"
    elif piece_count <= 18 and pawn_count >=10 :
        return "Endgame"
    else:
        return "Middlegame"

def classify_game(game_evaluations):
    #Balanced: Neither player had an advantage.
    #Sharp: A back and forth game where both players had chances
    #Sudden: A close game that was lost by the mistake
    #Smooth: One player took the advantage and never let go
    
    evaluation_diff = np.diff(game_evaluations)
    max_diff = np.max(np.abs(evaluation_diff))
    avg_diff = np.mean(np.abs(evaluation_diff))
    std_dev = np.std(game_evaluations)
    final_evaluation = game_evaluations[-1]

    print("Max_diff:",max_diff)
    print("avg_diff:",avg_diff)
    print("Std:",std_dev)

    if max_diff < 0.2 and avg_diff <= 0.1:
        return "Balanced"
    elif max_diff > 0.5 and avg_diff > 0.1 and std_dev > 0.2:
        return "Sharp"
    elif max_diff > 0.4 and avg_diff <= 0.1:
        return "Sudden"
    elif (final_evaluation > 0.7 and np.mean(game_evaluations[-5:]) > 0.7) or (final_evaluation < 0.3 and np.mean(game_evaluations[-5:]) < 0.3):
        return "Smooth"
    else:
        return "Sharp"
    
def determine_game_length(num_moves):
    if num_moves < 20:
        return "kısa"
    elif 20 <= num_moves < 40:
        return "orta"
    elif 40 <= num_moves < 60:
        return "uzun"
    else:
        return "çok uzun"
    
def classify_best_move_rate(rate):
    rate_percentage = rate * 100
    if rate_percentage < 30:
        return "kötü"
    elif 30 <= rate_percentage < 60:
        return "orta"
    elif 60 <= rate_percentage < 75:
        return "iyi"
    else:
        return "çok iyi"
    
def determine_game_end_reason(mate_value):
    if mate_value == "No forced mate detected":
        return "terk"
    return "Mat fark edip terk"