import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randint
import logging

# TODO: Consider increasing grid size?

GRID_X_SIZE = 32
GRID_Y_SIZE = 27
TOTAL_INPUT_FEATURES = GRID_X_SIZE*GRID_Y_SIZE + 2
TOTAL_OUTPUT_FEATURES = 2

ORANGE_PEG_WEIGHTING = 5
BLUE_PEG_WEIGHTING = 1
GREEN_PEG_WEIGHTING = 3
PURPLE_PEG_WEIGHTING = 2

# Inputs:
# Peg score - 32*27 grid, score based on pegs in cells. 5 for orange, 3 for green, 2 for pink, 1 for blue.
# Number of orange pegs left
# Number of blue pegs left
# Total: 866 inputs. 2 outputs.


# TERMINOLOGY
# Peg location - Location of peg with respect to coordinates of the full board
# Mapped peg location - Location of peg with respect to coordinates of the grid representation of the board as a 32x27 grid.
# Mapped peg board - Representation of the board as a 32x27 grid with locations of a given peg colour as ones.
# Scored board - Combination of all mapped peg boards, with each colour weighted differently and then all summed together.


# PeggleBrain inherits the torch module class from nn i.e. a basic neural network
class PeggleBrain(nn.Module):
    
    # Create a neural network with i, j, and k amounts of input, hidden, and output nodes
    def __init__(self, in_features=TOTAL_INPUT_FEATURES, h1=100, h2=100, out_features=TOTAL_OUTPUT_FEATURES):
        super().__init__() # performs base initialisation of nn.Module class.
        self.fc1 = nn.Linear(in_features, h1) # fc1 = fully connected 1. This is the connections from the nodes in the input layer to the first hidden layer.
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x
    
    
    def clean_peg_locations(self, peg_locations):
        
        if not np.any(peg_locations):
            return np.zeros((27, 32))
        
        # Convert peg locations to a NumPy array and map locations to 32 by 27 grid.
        peg_locations = np.array(peg_locations)
        peg_locations[:, 0] = np.ceil(peg_locations[:, 0]/GRID_X_SIZE)
        peg_locations[:, 1] = np.ceil(peg_locations[:, 1]/GRID_Y_SIZE)
        
        # Remove all duplicate coordinates.
        mapped_locations = np.unique(peg_locations, axis=0)
        
        # Create the 32 by 27 grid representation of the board.
        mapped_board = np.zeros((27, 32))
        for x, y in mapped_locations:
            mapped_board[y, x] = 1
            
        return mapped_board
    
    
    def get_board_with_scores(self, mapped_orange_peg_board, mapped_blue_peg_board):
        
        # Weight mapped peg boards by multiplying by weight.
        weighted_mapped_orange_peg_board = mapped_orange_peg_board*ORANGE_PEG_WEIGHTING
        weighted_mapped_blue_peg_board = mapped_blue_peg_board*BLUE_PEG_WEIGHTING
        
        # Combine mapped peg boards into one board.
        scored_board = weighted_mapped_orange_peg_board + weighted_mapped_blue_peg_board
        return scored_board
    
    
    def create_input_tensor(self, scored_board, n_orange, n_blue):
        scored_board_tensor = torch.flatten(torch.FloatTensor(scored_board))
        n_tensor = torch.FloatTensor([n_orange, n_blue])
        input_tensor = torch.cat([scored_board_tensor, n_tensor])
        return input_tensor
        
        
    
    def select_shot(self, orange_peg_locations, blue_peg_locations):
        # Take an input of orange peg and blue peg locations. Put it into neural network and predict x and y location.
        
        # Create mapped peg boards.
        mapped_orange_peg_board = self.clean_peg_locations(orange_peg_locations)
        mapped_blue_peg_board = self.clean_peg_locations(blue_peg_locations)
        
        # Calculate number of cells with pegs in them for each type. Pegs may be intersected between cells.
        n_orange = np.sum(mapped_orange_peg_board)
        n_blue = np.sum(mapped_blue_peg_board)
        
        # Get mapped board with cell scores.
        scored_board = self.get_board_with_scores(mapped_orange_peg_board, mapped_blue_peg_board)
        
        # Create input tensor and predict
        input_tensor = self.create_input_tensor(scored_board, n_orange, n_blue)
        x, y = self.forward(input_tensor)
        
        return x, y
        
        
    
class PeggleNaive:
    
    def __init__(self):
        pass
    

    def select_shot(self, locations):
        try:
            x, y = locations[0]
            return (x + randint(-15, 15), y + randint(-15, 15))
        except IndexError:
            logging.warning("No orange pegs found. Shooting at (10, 10).")
            return (10, 10)