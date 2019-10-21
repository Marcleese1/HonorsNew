# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 05:27:37 2019

@author: mjlee
"""

from enum import Enum
from enum import IntEnum
from random import *

full_deck = []
partial_deck = []


#Card enum for playing cards
class Card(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

#suit enum for playing cards
class Suit(Enum):
    SPADES = 'spades'
    CLUBS = 'clubs'
    DIAMONDS = 'diamonds'
    HEARTS = 'hearts'

#class to hold information for playing cards
class playing_card:
    def __init__(self, card_value, card_suit):
        self.card = card_value
        self.suit = card_suit
        

#function to deal full deck of cards    
def create_deck():
    for suit in Suit:
        for card in Card:
            full_deck.append(playing_card(Card(card), Suit(suit)))
    return full_deck

#draw single card from "deck"
def draw_card(deck):
    rand_card = randint(0,len(deck) -1)
    return deck.pop(rand_card)

create_deck()
partial_deck = list(full_deck)





