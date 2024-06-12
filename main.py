import pygame
from pygame.math import Vector2
from pieces.Bishop import Bishop
from pieces.King import King   
from pieces.Knight import Knight
from pieces.Pawn import Pawn
from pieces.Rock import Rock
from pieces.Queen import Queen 
from functools import reduce
import operator
import chess
import chess.engine
import tkinter as tk
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from read_data import pgn_to_dataframe,convertSan
import re
import numpy as np
from game_features import classify_game_phase, classify_game, classify_best_move_rate, determine_game_length, classify_best_move_rate, determine_game_end_reason
import pandas as pd
import requests
from bs4 import BeautifulSoup
from Levenshtein import distance as levenshtein_distance
from read_data import pgn_to_dataframe, convertSan

# Load the models

model = pickle.load(open('logmodel.pk', 'rb'))
vectorizer_model = pickle.load(open('vectorizer.pk', 'rb'))
onehotencoder = pickle.load(open('onehot.pk', 'rb'))

class Game():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess")
        self.board = chess.Board()
        self.moves = []
        self.cg_board = (720, 480)  # Enlarged board
        self.screen = pygame.display.set_mode(self.cg_board)
        self.gameObj = {"w": [], "b": []}
        self.setup()
        self.scene()
        self.running = True
        self.index = 0


        self.font = pygame.font.Font(None, 64)
        self.button = pygame.Rect(500, 60, 200, 50)  # Button for analysis
        self.message = "Hamle: "
        self.prediction_message = ""
        self.similar_message = ""
        self.player = None


    def coord2not(self, x, y):
        notation = "abcdefgh"
        return notation[x] + str(abs(8-y))
    
    def moves2eco(self):
        eco_df = pd.read_parquet('ECO.parquet')
        distances = []
        chess_moves_str = ' '.join(self.moves)

        for index, row in eco_df.iterrows():
            distance = levenshtein_distance(chess_moves_str, row['UCI Moves'])
            distances.append(distance)
        eco_df['Levenshtein Distance'] = distances
        
        #print(chess_moves_str)
        min_distance_row = eco_df.loc[eco_df['Levenshtein Distance'].idxmin()]

        if min_distance_row is not None:
            eco_code = min_distance_row['ECO Code']
            #print(f"Bu oyunun ECO kodu: {eco_code}")
            #print(f"Levenshtein Mesafesi: {min_distance_row['Levenshtein Distance']}")
            return eco_code
        else:
            print("Bu oyunun ECO kodu: A00")
            return 'A00'
            

    def game2vec(self):
        return vectorizer_model.transform([convertSan(self.moves)])
        # return self.moves
        # ['e2e4']

    def calculate_mean_similarity(self, gm_vector, game_vector):
        similarities = [cosine_similarity(gm_vector, vector)[0][0] for vector in game_vector]
        return sum(similarities) / len(similarities) if len(similarities) > 0 else 0
    
    def analysis(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-macos-m1-apple-silicon")
        self.analysis_board = chess.Board()
        best_move_count = 0
        best_move = chess.Move.from_uci('e2e4')
        game_info = {}
        wdl_expectations = []
        for move in self.moves:
            self.analysis_board.push(chess.Move.from_uci(move))
            if move == best_move:
                best_move_count += 1
            info = self.engine.analyse(self.analysis_board, chess.engine.Limit(time=0.1))
            try:
                best_move = info["pv"][0]
            except KeyError:
                #print('That was a last move.')
                pass
            wdl_expectation = info["score"].white().wdl().expectation()
            wdl_expectations.append(wdl_expectation)

        mate_in = info["score"].relative.mate()
        if mate_in is not None:
            game_info['Mate'] = abs(mate_in)
        else:
            game_info['Mate'] = "No forced mate detected" 
        
        piece_count = len(self.analysis_board.piece_map())
        move_count = len(self.moves)
        pawn_count = sum(1 for piece in self.analysis_board.piece_map().values() if piece.piece_type == chess.PAWN)

        game_phase = classify_game_phase(move_count, piece_count, pawn_count)
        best_move_rate = best_move_count / move_count if move_count > 0 else 0
        class_of_game = classify_game(wdl_expectations)

        game_info['ECO'] = self.moves2eco()
        game_info['Phase'] = game_phase
        game_info['Best Move Rate'] = best_move_rate
        game_info['Game Class'] = class_of_game
        game_info['Number of Moves'] = move_count
        game_info['Game Length'] = determine_game_length(move_count)
        game_info['Best Move Rate Classify'] = classify_best_move_rate(best_move_rate)
        game_info['Game Ending Reason'] = determine_game_end_reason(mate_in)
        
        #print(game_info)
        game = pd.DataFrame([game_info])
        X = onehotencoder.transform(game[["ECO", "Game Class", "Phase", "Game Length", "Best Move Rate Classify", "Game Ending Reason"]])

        self.pred = model.predict(X)[0]

        self.engine.quit()
        
        if self.pred == 0:
            self.player = 'Carlsen'
        elif self.pred == 1:
            self.player = 'Nakamura'
        elif self.pred == 2:
            self.player = 'Caruana'
        
        self.prediction_message = f"Prediction: {self.player}"

        max_similars = self.similar_games()

        self.similar_message = [
            f"Carlsen similarity: {max_similars[0] * 100:.2f}%",
            f"Nakamura similarity: {max_similars[1] * 100:.2f}%",
            f"Caruana similarity: {max_similars[2] * 100:.2f}%"
        ]


    
    def similar_games(self):
        similarity_rate = []
        for player in ['Carlsen', 'Nakamura', 'Caruana']:
            player_games = pd.read_parquet(f"{player} V2.parquet")
            game_vector = vectorizer_model.transform(player_games['Moves'])
            gui_game = convertSan(self.moves)
            gui_vector = vectorizer_model.transform([gui_game])

            similarities = cosine_similarity(gui_vector, game_vector)
            max_similarity = similarities.max()
            similarity_rate.append(max_similarity)
        return similarity_rate

    def update_move_label(self, move):
        self.moves.append(move)
        self.message = f"Hamle: {move}"

    def setup(self):
        self.background = pygame.image.load("images/chessBoard.png")
        self.gameObj = self.fen_to_board(self.board.fen())
    
    def pieceState(self, pi=None):
        for p in reduce(operator.concat, self.gameObj.values()):
            if p == pi:
                self.screen.blit(pi.image, pi.rect)
            else:
                self.screen.blit(p.image, p.getPosition())
        pygame.display.update()

    def scene(self):
        self.background = pygame.transform.scale(self.background, (480, 480))  # Adjust to fit larger screen
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        self.pieceState()
        pygame.display.update()

    def fen_to_board(self, fen):
        Obj = {"w": [], "b": []}
        blank = 0
        for row in fen.split('/'):
            index = 0
            for c in row:
                if c == ' ':
                    break
                elif c in '12345678':
                    index += int(c) - 1
                elif c == 'p':
                    Obj["b"].append(Pawn("b", (index, blank)))
                elif c == 'P':
                    Obj["w"].append(Pawn("w", (index, blank)))
                elif c == 'r':
                    Obj["b"].append(Rock("b", (index, blank)))
                elif c == 'R':
                    Obj["w"].append(Rock("w", (index, blank)))
                elif c == 'n':
                    Obj["b"].append(Knight("b", (index, blank)))
                elif c == 'N':
                    Obj["w"].append(Knight("w", (index, blank)))
                elif c == 'b':
                    Obj["b"].append(Bishop("b", (index, blank)))
                elif c == 'B':
                    Obj["w"].append(Bishop("w", (index, blank)))
                elif c == 'q':
                    Obj["b"].append(Queen("b", (index, blank)))
                elif c == 'Q':
                    Obj["w"].append(Queen("w", (index, blank)))
                elif c == 'k':
                    Obj["b"].append(King("b", (index, blank)))
                elif c == 'K':
                    Obj["w"].append(King("w", (index, blank)))
                index += 1
            blank += 1
        return Obj
    
    def update(self, piece):
        self.screen.blit(self.background, (0, 0))
        self.pieceState(piece)
        self.draw_button()      

        # Message display
        self.display_message()  
        self.display_prediction_message() 
        self.display_similar_message() 

        pygame.display.update()

    def draw_button(self):
        pygame.draw.rect(self.screen, [0, 128, 0], self.button)
        text_surf = self.font.render('Analiz Et', True, [255, 255, 255])
        self.screen.blit(text_surf, self.button)

    def display_message(self):
        #text_surf = self.font.render(self.message, True, [255, 255, 255])
        #self.screen.blit(text_surf, (480, 200))
        pygame.draw.rect(self.screen, (0, 0, 0), (480, 200, 240, 50))
        text_surf = pygame.font.Font(None, 40).render(self.message, True, [255, 255, 255])
        self.screen.blit(text_surf, (480, 200))

    def display_prediction_message(self):
        # Clear the previous prediction message
        pygame.draw.rect(self.screen, (0, 0, 0), (480, 300, 240, 50))
        text_surf = pygame.font.Font(None, 24).render(self.prediction_message, True, [255, 255, 255])
        self.screen.blit(text_surf, (480, 300))

    def display_similar_message(self):
        # Clear the previous prediction message
        pygame.draw.rect(self.screen, (0, 0, 0), (480, 400, 240, 60))
        #text_surf = pygame.font.Font(None, 24).render(self.similar_message, True, [255, 255, 255])
        #self.screen.blit(text_surf, (480, 400))

        for i, line in enumerate(self.similar_message):
            text_surf = pygame.font.Font(None, 24).render(line, True, [255, 255, 255])
            self.screen.blit(text_surf, (480, 400 + i * 20))  

    def main(self):
        moving = False
        firstCoordinate = None
        p = None
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif self.board.is_checkmate():
                    self.analysis()
                    #self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.button.collidepoint(event.pos):
                        self.analysis()
                    elif self.index % 2 == 0:
                        for _ in self.gameObj["w"]: 
                            if _.rect.collidepoint(event.pos):     
                                moving = True
                                firstCoordinate = _.coord2not()
                                p = _
                    else:
                        for _ in self.gameObj["b"]: 
                            if _.rect.collidepoint(event.pos):     
                                moving = True
                                firstCoordinate = _.coord2not()
                                p = _
                elif event.type == pygame.MOUSEBUTTONUP:
                    moving = False
                    if p is not None:
                        x, y = pygame.mouse.get_pos()
                        try:
                            if chess.Move.from_uci(firstCoordinate + self.coord2not(x // 60, y // 60)) in self.board.legal_moves:
                                p.position = Vector2(x // 60, y // 60)
                                self.board.push(chess.Move.from_uci(firstCoordinate + self.coord2not(x // 60, y // 60)))
                                self.gameObj = self.fen_to_board(self.board.fen())
                                self.index += 1
                                self.update_move_label(firstCoordinate + self.coord2not(x // 60, y // 60))
                            else:
                                print("illegal")
                        except chess.InvalidMoveError:
                            print("invalid")
                        p.rectUpdate()
                        p = None
                elif event.type == pygame.MOUSEMOTION and moving:
                    p.rect.move_ip(event.rel)
            self.update(p)
            pygame.display.flip()

if __name__ == "__main__":
    game = Game()
    game.main()
