import numpy as np
import random
import pickle
import os
PLAYER_SYMBOL = 'X'
AI_SYMBOL = 'O'
EMPTY = 0

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
    def get_state_key(self, board):
        return str(board.flatten())
    
    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in valid_actions}
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [action for action, q_val in q_values.items() if q_val == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state) if next_state is not None else None
        if next_state_key and next_state_key not in self.q_table:
            valid_next_actions = [(i // 3, i % 3) for i in range(9) if next_state.flatten()[i] == EMPTY]
            self.q_table[next_state_key] = {action: 0.0 for action in valid_next_actions}
        
        # Calculate max Q for next state
        next_max = max(self.q_table[next_state_key].values()) if next_state_key and next_state_key in self.q_table else 0
        
        # Q-learning update
        old_value = self.q_table[state_key][action]
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_key][action] = new_value
    
    def save_q_table(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename='q_table.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
            return True
        return False

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player is 1, AI is 2
        self.game_over = False
        self.winner = None
        self.q_agent = QLearningAgent()
        self.ai_enabled = True
        self.training_mode = False
        self.total_games = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.ties = 0
        
        # Try to load pre-trained Q-table
        if not self.q_agent.load_q_table():
            print("No pre-trained model found. Training a new one...")
            self.train_ai(10000)
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        
    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == EMPTY]
    
    def make_move(self, row, col):
        if self.game_over or self.board[row, col] != EMPTY:
            return False
        
        self.board[row, col] = self.current_player
        self.check_game_state()
        
        # Switch player if game is not over
        if not self.game_over:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            
            # AI's turn
            if self.current_player == 2 and self.ai_enabled and not self.game_over:
                self.ai_move()
        
        return True
    
    def ai_move(self):
        if self.game_over:
            return
        
        valid_actions = self.get_valid_moves()
        
        if valid_actions:
            state_key = self.q_agent.get_state_key(self.board)
            
            if state_key in self.q_agent.q_table:
                q_values = self.q_agent.q_table[state_key]
                max_q = -float('inf')  # Initialize with negative infinity
                best_actions = []

                for action in valid_actions:
                    if action in q_values and q_values[action] > max_q:
                        max_q = q_values[action]
                        best_actions = [action]
                    elif action in q_values and q_values[action] == max_q:
                        best_actions.append(action)

                if best_actions:
                    action = random.choice(best_actions)
                else:
                    action = random.choice(valid_actions) # Fallback if no valid action in Q-table
            else:
                action = random.choice(valid_actions) # Fallback to random if state not in Q-table
            
            row, col = action
            self.board[row, col] = self.current_player
            self.check_game_state()
            
            if not self.game_over:
                self.current_player = 3 - self.current_player
    
    def check_game_state(self):
        # Check rows
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != EMPTY:
                self.game_over = True
                self.winner = self.board[i, 0]
                return
        
        # Check columns
        for j in range(3):
            if self.board[0, j] == self.board[1, j] == self.board[2, j] != EMPTY:
                self.game_over = True
                self.winner = self.board[0, j]
                return
        
        # Check diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != EMPTY:
            self.game_over = True
            self.winner = self.board[0, 0]
            return
        
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != EMPTY:
            self.game_over = True
            self.winner = self.board[0, 2]
            return
        
        # Check for tie
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
            return
    
    def train_ai(self, episodes=1000):
        print(f"Training AI with {episodes} episodes...")
        self.training_mode = True
        
        for episode in range(episodes):
            self.reset()
            state = self.board.copy()
            
            while not self.game_over:
                valid_actions = self.get_valid_moves()
                if not valid_actions: # No valid moves, it's a tie before any more moves can be made
                    self.game_over = True
                    self.winner = 0
                    break

                action = self.q_agent.choose_action(state, valid_actions)
                row, col = action
                
                # Make move
                self.board[row, col] = self.current_player
                self.check_game_state()
                next_state = self.board.copy()
                
                # Determine reward
                reward = 0
                if self.game_over:
                    if self.winner == 2:  # AI wins
                        reward = 1
                    elif self.winner == 1:  # Player wins (opponent wins)
                        reward = -1
                    else:  # Tie
                        reward = 0
                
                # Update Q-value (from the perspective of the current player making the move)
                self.q_agent.update_q_value(state, action, reward, next_state)
                
                # Switch player
                self.current_player = 3 - self.current_player
                state = next_state
            
            # Update stats
            if self.winner == 1:
                self.player_wins += 1
            elif self.winner == 2:
                self.ai_wins += 1
            else:
                self.ties += 1
            self.total_games += 1
            
            if (episode + 1) % (episodes // 10) == 0:
                print(f"  Completed {episode + 1}/{episodes} episodes.")
        
        self.q_agent.save_q_table()
        self.training_mode = False
        print("Training completed!")

    def print_board(self):
        symbols = {0: ' ', 1: PLAYER_SYMBOL, 2: AI_SYMBOL}
        print("\n-------------")
        for r in range(3):
            print(f"| {symbols[self.board[r, 0]]} | {symbols[self.board[r, 1]]} | {symbols[self.board[r, 2]]} |")
            print("-------------")

    def play_game(self):
        self.reset()
        print("\n--- New Game Started ---")
        self.print_board()

        while not self.game_over:
            if self.current_player == 1:
                print("Your turn (Player X):")
                valid_move = False
                while not valid_move:
                    try:
                        row = int(input("Enter row (0-2): "))
                        col = int(input("Enter column (0-2): "))
                        if 0 <= row <= 2 and 0 <= col <= 2:
                            if self.make_move(row, col):
                                valid_move = True
                            else:
                                print("Invalid move. Cell already taken or game over. Try again.")
                        else:
                            print("Invalid input. Row and column must be between 0 and 2.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            else: # AI's turn
                print("AI's turn (Player O)...")
                self.ai_move()
            
            self.print_board()

            if self.game_over:
                if self.winner == 1:
                    print("Player X wins!")
                elif self.winner == 2:
                    print("AI (O) wins!")
                else:
                    print("It's a tie!")
                break

        self.total_games += 1
