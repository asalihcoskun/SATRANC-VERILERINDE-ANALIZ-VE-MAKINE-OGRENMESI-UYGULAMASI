import pygame
from pygame.math import Vector2


class Piece():
    def __init__(self,color,position):
        self.color = color
        self.kill = False
        self.path = "images/"
        self.position = Vector2(position)
        self.rect = pygame.Rect(0,0,60,60)
        self.rectUpdate()
        
    def getPosition(self):
        px = self.position.elementwise() * Vector2(60,60)
        return ((int(px.x),int(px.y)))

    def getCoordinate(self):
        x,y = self.getPosition()
        #self.position = Vector2(x,y)
        return (x//60,y//60)
    
    def getNotation(self):
        pass

    def rectUpdate(self):
        self.rect.x = self.getPosition()[0]
        self.rect.y = self.getPosition()[1]
    
    def coord2not(self):
        notation = "abcdefgh"
        x,y = self.getCoordinate()
        return notation[x] + str(abs(8-y))

    def notation(self,pos):
        x,y = pos
        notation = "abcdefgh"
        return notation[x] + str(abs(8-y))

    def __str__(self):
        return self.__class__.__name__