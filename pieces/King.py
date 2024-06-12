import pygame
from pieces.Piece import Piece

class King(Piece):
    def __init__(self,color,position):
        super().__init__(color,position)
        self.image = pygame.image.load(self.path + str.lower(self.__class__.__name__[0])+self.color + ".png")