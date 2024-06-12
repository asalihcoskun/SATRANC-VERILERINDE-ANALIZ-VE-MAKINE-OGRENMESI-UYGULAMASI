import chess.pgn
import pandas as pd
import re

def pgn_to_dataframe(file_path):
    games = []
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            headers = game.headers
            result = headers.get("Result")
            headers["Result"] = result.split("-") if result else None
            data = {
                "White": headers.get("White"),
                "Black": headers.get("Black"),
                "ECO": headers.get("ECO"),
                "Opening": headers.get("Opening"),
                "TimeControl": headers.get("TimeControl"),
                "Termination": headers.get("Termination"),
                "Result": headers.get("Result"),
                "Moves": [move.uci() for move in game.mainline_moves()]
            }
            games.append(data)
    df = pd.DataFrame(games)
    return df

def convertSan(moves):
    board = chess.Board()
    move_list = board.variation_san([chess.Move.from_uci(m) for m in moves])
    moves = re.findall(r'\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|\d+-\d+|#)\b', move_list)
    return " ".join(moves)

