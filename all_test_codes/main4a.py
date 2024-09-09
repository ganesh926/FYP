import pygame
import sys
import json
from collections import deque
import random
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 700
NODE_RADIUS = 15
FPS = 60

# Colors
BACKGROUND = (245, 222, 179)  # Light wooden color
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # For tigers
GREEN = (0, 128, 0)  # For goats
BLUE = (0, 0, 255)  # For highlighting
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Aadu Puli Aattam (Goats and Tigers)")

# Node positions (update these to match the traditional game board layout)
node_positions = {
    0: (400, 100),
    1: (100, 200), 2: (320, 200), 3: (390, 200), 4: (440, 200), 5: (500, 200), 6: (700, 200),
    7: (80, 300), 8: (250, 300), 9: (380, 300), 10: (470, 300), 11: (580, 300), 12: (730, 300),
    13: (60, 400), 14: (180, 400), 15: (370, 400), 16: (500, 400), 17: (660, 400), 18: (760, 400),
    19: (120, 500), 20: (360, 500), 21: (530, 500), 22: (740, 500)
}

# Connections between nodes
connections = [
   (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12),
    (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (7, 13), (8, 14), (9, 15), (10, 16), (11, 17), (12, 18),
    (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    (14, 19), (15, 20), (16, 21), (17, 22),
    (19, 20), (20, 21), (21, 22)
]
capture_connections = [
    [8, 9, 10, 11],  # 0
    [3, 13],         # 1
    [4, 14],         # 2
    [1, 5, 15],      # 3
    [2, 6, 16],      # 4
    [3, 17],         # 5
    [4, 18],         # 6
    [9],             # 7
    [0, 10, 19],     # 8
    [0, 7, 11, 20],  # 9
    [0, 8, 12, 21],  # 10
    [0, 9, 22],      # 11
    [10],            # 12
    [1, 15],         # 13
    [2, 16],         # 14
    [3, 13, 17],     # 15
    [4, 14, 18],     # 16
    [5, 15],         # 17
    [6, 16],         # 18
    [8, 21],         # 19
    [9, 22],         # 20
    [10, 19],        # 21
    [11, 20]        # 22
]
move_connections = [
    [2, 3, 4, 5], #0
    [2, 7], #1
    [0, 1, 3, 8], #2
    [0, 2, 4, 9], #3
    [0, 3, 5, 10], #4
    [0, 4, 6, 11], #5
    [5, 12], #6
    [1, 8, 13], #7
    [2, 7, 9, 14], #8
    [3, 8, 10, 15], #9
    [4, 9, 11, 16], #10
    [5, 10, 12, 17], #11
    [6, 11, 18], #12
    [7, 14], #13
    [8, 13, 15, 19], #14
    [9, 14, 16, 20], #15
    [10, 15, 17, 21], #16
    [11, 16, 18, 22], #17
    [12, 17], #18
    [14, 20], #19
    [15, 19, 21], #20
    [16, 20, 22], #21
    [17, 21] #22
    ]

capture_node={
    (0, 8): 2, (0, 9): 3, (0, 10): 4, (0, 11): 5,
    (1, 3): 2, (1, 13): 7,
    (2, 4): 3, (2, 14): 8,
    (3, 1): 2, (3, 5): 4, (3, 15): 9,
    (4, 2): 3, (4, 6): 5, (4, 16): 10,
    (5, 3): 4, (5, 17): 11,
    (6, 4): 5, (6, 18): 12,
    (7, 9): 8,
    (8, 0): 2, (8, 10): 9, (8, 19): 14,
    (9, 0): 3, (9, 7): 8, (9, 11): 10, (9, 20): 15,
    (10, 0): 4, (10, 8): 9, (10, 12): 11, (10, 21): 12,
    (11, 0): 5, (11, 9): 10, (11, 22): 17,
    (12, 10): 11,
    (13, 1): 7, (13, 15): 14,
    (14, 2): 8, (14, 16): 15,
    (15, 3): 9, (15, 13): 14, (15, 17): 16,
    (16, 4): 10, (16, 13): 15, (16, 18): 17,
    (17, 5): 11, (17, 15): 16,
    (18, 6): 12, (18, 16): 17,
    (19, 8): 14, (19, 21): 20,
    (20, 9): 15, (20, 22): 21,
    (21, 10): 16, (21, 19): 20,
    (22, 11): 17, (22, 20): 21
}
# Game state
pieces = {'tigers': [0], 'goats': []}
current_player = 'goat'
goats_to_place = 15
selected_node = None
captured_goats = 0
tiger_wins = 0
goat_wins = 0
move_count = 0
game_over = False
tiger_agent=None

# Constants for scrolling
SCROLL_BAR_WIDTH = 20
SCROLL_STEP = 18  # Height of each move entry
scroll_position = 0  # Initial scroll position
max_visible_moves = 3  # Number of moves visible at a time
moves = []
options = ['Goat vs Tiger', 'Goat vs AI', 'Goat vs Bot', 'Bot vs AI']
selected_index=0
game_mode = options[0]


# Define the state and action spaces
STATE_SPACE_SIZE = len(node_positions)
ACTION_SPACE_SIZE = len(node_positions)


class TigerHeuristicBot:
    def __init__(self, capture_connections, connections, capture_node):
        self.capture_connections = capture_connections
        self.connections = connections
        self.capture_node = capture_node

    def is_adjacent(self, node1, node2):
        return (node1, node2) in self.connections or (node2, node1) in self.connections

    def get_capture_moves(self, tiger, pieces):
        possible_moves = []
        for end in self.capture_connections[tiger]:
            mid = self.capture_node.get((tiger, end))
            if mid and mid in pieces['goats'] and end not in pieces['tigers'] and end not in pieces['goats']:
                possible_moves.append((tiger, end))
        return possible_moves

    def get_regular_moves(self, tiger, pieces):
        possible_moves = []
        for connection in self.connections:
            if tiger in connection:
                adjacent_node = connection[1] if connection[0] == tiger else connection[0]
                if adjacent_node not in pieces['tigers'] and adjacent_node not in pieces['goats']:
                    possible_moves.append((tiger, adjacent_node))
        return possible_moves

    def evaluate_move(self, move, pieces):
        start, end = move
        score = 0

        # Prioritize moves away from goats
        goat_positions = pieces['goats']
        if end not in goat_positions:
            score += 50

        # Prioritize moves towards the center
        center_nodes = [0, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17]
        if end in center_nodes:
            score += 30

        # Penalize moves that block goats
        blocked_goats = sum(1 for adj in self.capture_connections[end] if adj in goat_positions)
        score -= blocked_goats * 20

        # Penalize moves that create future capture opportunities
        future_captures = sum(1 for adj in self.capture_connections[end] if adj not in goat_positions and adj not in pieces['tigers'])
        score -= future_captures * 15

        # Prioritize moves that put tigers in danger
        for goat in goat_positions:
            if self.is_adjacent(end, goat):
                score += 5

        return score

    def get_all_nodes(self):
        return list(set([node for connection in self.connections for node in connection]))

    def get_placement_move(self, pieces):
        tiger_positions = pieces['tigers']
        goat_positions = pieces['goats']
        available_nodes = set(self.get_all_nodes()) - set(tiger_positions) - set(goat_positions)

        def evaluate_placement(node):
            score = 0
            # Prefer positions away from goats
            if node not in goat_positions:
                score += 30

            # Prefer central positions
            center_nodes = [0, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17]
            if node in center_nodes:
                score += 20

            return score

        return max(available_nodes, key=evaluate_placement)

    def get_best_move(self, pieces):
        all_moves = []
        for tiger in pieces['tigers']:
            all_moves.extend(self.get_capture_moves(tiger, pieces))
            all_moves.extend(self.get_regular_moves(tiger, pieces))
        
        if not all_moves:
            return None

        # Evaluate and sort moves
        evaluated_moves = [(move, self.evaluate_move(move, pieces)) for move in all_moves]
        evaluated_moves.sort(key=lambda x: x[1], reverse=True)

        # Return the best move
        return evaluated_moves[0][0]

    def are_tigers_trapped(self, pieces):
        for tiger in pieces['tigers']:
            adjacent_nodes = self.connections[tiger]
            goat_positions = pieces['goats']
            if any(node in goat_positions for node in adjacent_nodes):
                return False
        return True


class GoatHeuristicBot:
    def __init__(self, connections, capture_connections, capture_node):
        self.connections = connections
        self.capture_connections = capture_connections
        self.capture_node = capture_node

    def get_all_nodes(self):
        return list(set([node for connection in self.connections for node in connection]))

    def is_adjacent(self, node1, node2):
        return (node1, node2) in self.connections or (node2, node1) in self.connections

    def evaluate_placement(self, node, pieces):
        score = 0
        
        # Strongly prioritize nodes that contribute to trapping tigers
        trap_contribution = self.evaluate_trap_contribution(node, pieces)
        score += trap_contribution * 300
        
        # Avoid nodes that can be immediately captured
        for tiger in pieces['tigers']:
            if node in self.capture_connections[tiger]:
                mid = self.capture_node.get((tiger, node))
                if mid and mid not in pieces['goats'] and mid not in pieces['tigers']:
                    score -= 500  # Penalize potential capture positions
        
        # Prefer central positions
        central_nodes = [0, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17]
        if node in central_nodes:
            score += 20
    
        return score

    def evaluate_move(self, start, end, pieces):
        score = 0
        
        # Strongly prioritize moves that contribute to trapping tigers
        trap_contribution = self.evaluate_trap_contribution(end, pieces)
        score += trap_contribution * 300
        
        # Avoid moves that allow captures
        for tiger in pieces['tigers']:
            if end in self.capture_connections[tiger]:
                mid = self.capture_node.get((tiger, end))
                if mid and mid in pieces['goats']:
                    score -= 500  # Heavy penalty for moving into capture position
        
        # Prefer moves that block tigers
        tiger_adjacent_count = sum(1 for tiger in pieces['tigers'] if self.is_adjacent(end, tiger))
        score += tiger_adjacent_count * 40
    
        # Prioritize central positions
        central_nodes = [0, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17]
        if end in central_nodes:
            score += 20
    
        return score

    def evaluate_trap_contribution(self, node, pieces):
        trap_count = 0
        for tiger in pieces['tigers']:
            adjacent_nodes = [n for n in self.connections[tiger] if n != node]
            if all(n in pieces['goats'] or n == node for n in adjacent_nodes):
                trap_count += 1
        return trap_count

    def get_best_placement(self, pieces):
        available_nodes = set(self.get_all_nodes()) - set(pieces['tigers']) - set(pieces['goats'])
        return max(available_nodes, key=lambda node: self.evaluate_placement(node, pieces))

    def get_best_move(self, moves, pieces):
        best_move = None
        best_score = float('-inf')
        for move in moves:
            score = self.evaluate_move(move[0], move[1], pieces)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def make_move(self, pieces, goats_to_place):
        if goats_to_place > 0:
            return (None, self.get_best_placement(pieces))
        else:
            # Prioritize moves that contribute to trapping tigers
            trap_moves = []
            regular_moves = []
            for goat in pieces['goats']:
                for connection in self.connections:
                    if goat in connection:
                        end = connection[1] if connection[0] == goat else connection[0]
                        if end not in pieces['tigers'] and end not in pieces['goats']:
                            if self.evaluate_trap_contribution(end, pieces) > 0:
                                trap_moves.append((goat, end))
                            else:
                                regular_moves.append((goat, end))
        
            # First try trap moves, then regular moves
            if trap_moves:
                return self.get_best_move(trap_moves, pieces)
            else:
                return self.get_best_move(regular_moves, pieces)

    def is_tiger_trapped(self, tiger, pieces):
        adjacent_nodes = self.connections[tiger]
        return all(node in pieces['goats'] for node in adjacent_nodes)

    def are_tigers_trapped(self, pieces):
        for tiger in pieces['tigers']:
            if not self.is_tiger_trapped(tiger, pieces):
                return False
        return True


class TigerAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.01  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.replay_buffer = deque(maxlen=1000)
        self.state_history = []

    def get_all_nodes(self):
        # Get nodes from node_positions
        nodes_from_positions = set(node_positions.keys())

        # Get unique nodes from connections
        nodes_from_connections = set(connection for pair in connections for connection in pair)

        # Combine all nodes
        all_nodes = nodes_from_positions.union(nodes_from_connections)

        return list(all_nodes)
    
    def get_all_capture_moves_ai(self,tiger,pieces):
        possible_moves = []
        
        # Get all connections involving the tiger
        first_connections = [conn for conn in connections if tiger in conn]


        for start, end in first_connections:
            second_connections = [conn for conn in connections if end in conn]
            for start1, end1 in second_connections:

            
                mid = capture_node.get((start, end1),"")
                
                if mid and is_adjacent(start, mid) and is_adjacent(mid, end1):
                    if mid in pieces['goats'] and end1 not in pieces['tigers'] and end1 not in pieces['goats']:
                        possible_moves.append((start,end1))

        return possible_moves

    def get_placement_move(self, state):
        tiger_positions = state['tigers']
        goat_positions = state['goats']
        available_nodes = set(self.get_all_nodes()) - set(tiger_positions) - set(goat_positions)

        if random.random() < self.epsilon:
            return random.choice(list(available_nodes))
        else:
            return max(list(available_nodes), key=lambda x: self.q_table.get(x, 0))



    def get_action(self, state):
        if (len(pieces['tigers'])) < 3:
            node=tiger_agent.get_placement_move(state)
            return (None,node)
        elif random.random() < self.epsilon:
           
            return random.choice(self.get_possible_actions(state))
        else:
            return max(self.get_possible_actions(state), key=lambda x: self.q_table.get(x, 0))


  

    def get_possible_actions(self, state):
        tiger_moves = []
        tiger_pieces = state['tigers']
        goat_pieces = state['goats']

        for tiger in tiger_pieces:
            tiger_moves.append(self.get_all_capture_moves_ai(tiger, state))
        
        for tiger in tiger_pieces:
            for connection in connections:
                if tiger in connection:
                    adjacent_nodes = [conn[1] if conn[0] == tiger else conn[0] for conn in connections if tiger in conn]
                    
                    # If the tiger has no adjacent nodes, it cannot move
                    if adjacent_nodes:
                    
                    # Check if any adjacent node is free
                        for node in adjacent_nodes:
                            if node not in tiger_pieces and node not in goat_pieces:
                                tiger_moves.append((tiger, node))


        flat_list = [item for sublist in tiger_moves for item in (sublist if isinstance(sublist, list) else [sublist])]
        unique_list = list(set(flat_list))

        return unique_list

    def get_state_representation(self, pieces):
        state_representation = []
        for piece_type, positions in pieces.items():
            if isinstance(positions, int):
                positions = [positions]  # Handle if positions is an int
            elif not isinstance(positions, list):
                raise ValueError("Positions should be a list or int.")
            
            state_representation.extend((piece_type, piece_position) for piece_position in positions)
        
        return tuple(sorted(state_representation))

    def update_q_table(self, state, action, next_state, reward):
        # Check if q_table exists and is empty
        if not hasattr(self, 'q_table'):
            self.q_table = {}
        
        # Get current state representation
        current_state = self.get_state_representation(state)
        
        # Get next state representation
        next_state_rep = self.get_state_representation(next_state)
        
        # Create action tuple
        action_tuple = (current_state, action)
        
        # Check if the action tuple exists in q_table
        if action_tuple not in self.q_table:
            # Initialize Q-value for this state-action pair
            self.q_table[action_tuple] = 0
        
        # Update Q-value
        self.q_table[action_tuple] += self.alpha * (reward + self.gamma * max(self.q_table.values()) - self.q_table[action_tuple])
        
        # Store the next state representation
        self.next_state_rep = next_state_rep

    def save_q_table(self):
        with open('q_table.json', 'w') as f:
            json.dump({str(k): v for k, v in self.q_table.items()}, f)

    def load_q_table(self):
        try:
            with open('q_table.json', 'r') as file:
                loaded_q_table = json.load(file)
                self.q_table = {}
                for key, value in loaded_q_table.items():
                    try:
                        key_tuple = tuple(eval(key))
                        self.q_table[key_tuple] = value
                    except (ValueError, SyntaxError):
                        print(f"Skipping invalid key: {key}")
        except FileNotFoundError:
            print("No Q-table found, initializing a new one.")



def get_node_at_pos(pos):
    for node, node_pos in node_positions.items():
        distance = ((pos[0] - node_pos[0]) ** 2 + (pos[1] - node_pos[1]) ** 2) ** 0.5
        if distance < NODE_RADIUS + 5:
            return node
    return None

def is_adjacent(node1, node2):
    return (node1, node2) in connections or (node2, node1) in connections

def is_valid_move(start, end):
    return is_adjacent(start, end) and end not in pieces['tigers'] and end not in pieces['goats']


def is_capture_move(start_node, end_node):
    if start_node in pieces['tigers']:
        capture_connections_for_start = capture_connections[start_node]
        if end_node in capture_connections_for_start:
            intermediate_node = capture_node.get((start_node, end_node))
            if intermediate_node in pieces['goats'] and end_node not in pieces['tigers'] and end_node not in pieces['goats']:
                return intermediate_node
    return None

def get_all_capture_moves(tiger):
    possible_moves = []
    for end_node in range(len(node_positions)):
        captured_node = is_capture_move(tiger, end_node)
        if captured_node is not None:
            possible_moves.append((tiger, end_node))
    return possible_moves

def tiger_can_move(tiger):
    for adjacent_node in move_connections[tiger]:
        if adjacent_node not in pieces['tigers'] and adjacent_node not in pieces['goats']:
            return True
    return False

def are_tigers_trapped():
    for tiger in pieces['tigers']:
        if tiger_can_move(tiger) or get_all_capture_moves(tiger):
            return False
    return True


def reset_game():
    global pieces, current_player, goats_to_place, selected_node, captured_goats, game_over, moves, move_count
    pieces = {'tigers': [0], 'goats': []}
    current_player = 'goat'
    goats_to_place = 15
    selected_node = None
    captured_goats = 0
    game_over = False
    moves = []
    move_count = 0

def draw_board():
    screen.fill(BACKGROUND)
    for connection in connections:
        pygame.draw.line(screen, BLACK, node_positions[connection[0]], node_positions[connection[1]], 2)
    for node in node_positions:
        color = BLUE if node == selected_node else BLACK
        pygame.draw.circle(screen, color, node_positions[node], NODE_RADIUS)

def draw_pieces():
    for tiger in pieces['tigers']:
        pygame.draw.circle(screen, RED, node_positions[tiger], NODE_RADIUS - 2)
    for goat in pieces['goats']:
        pygame.draw.circle(screen, GREEN, node_positions[goat], NODE_RADIUS - 2)

def combobox_handling(event):
    global game_mode, selected_index, options, move_count
    
    if event.type == pygame.MOUSEBUTTONDOWN:
        combo_rect = pygame.Rect(20, 60, 200, 40)
        if combo_rect.collidepoint(event.pos) and move_count == 0:
            selected_index = (selected_index + 1) % len(options)
            game_mode = options[selected_index]
            draw_ui()  # Update UI immediately after selection

def draw_ui():
    global scroll_position, selected_index, move_count

    # Draw the background for the combo box
    combo_rect = pygame.Rect(20, 60, 200, 40)
    pygame.draw.rect(screen, WHITE, combo_rect)
    pygame.draw.rect(screen, BLACK, combo_rect, 2)

    # Render the selected option text
    font = pygame.font.Font(None, 32)
    text = font.render(options[selected_index], True, BLACK)
    screen.blit(text, (combo_rect.x + 10, combo_rect.y + 10))

    # Draw the history box
    history_box_y = HEIGHT - 130
    pygame.draw.rect(screen, WHITE, (20, history_box_y, WIDTH - 40, 90))
    pygame.draw.rect(screen, BLACK, (20, history_box_y, WIDTH - 40, 90), 2)

    # Scrollable move history
    move_font = pygame.font.Font(None, 22)
    total_moves = len(moves)
    max_scroll_position = max(0, total_moves - max_visible_moves)+1



    # Render visible moves

    for i in range(max_visible_moves):
        if scroll_position + i < total_moves:
            move_text = move_font.render(moves[scroll_position + i], True, BLACK)
            screen.blit(move_text, (30, history_box_y + 10 + i * SCROLL_STEP))
    
    # Draw the scroll bar
    if total_moves > max_visible_moves:
        scroll_bar_x = WIDTH - 35
        scroll_bar_height = max(20, 90 * max_visible_moves / total_moves)
        max_scroll_position = max(0, total_moves - max_visible_moves)+1
        scroll_thumb_y = history_box_y + (scroll_position / max_scroll_position) * (90 - scroll_bar_height)
        pygame.draw.rect(screen, BLACK, (scroll_bar_x, history_box_y, SCROLL_BAR_WIDTH, 90), 2)
        pygame.draw.rect(screen, GRAY, (scroll_bar_x + 2, scroll_thumb_y, SCROLL_BAR_WIDTH - 4, scroll_bar_height))
    
    # Displaying score and current player
    tiger_text = font.render(f"Tiger Wins: {tiger_wins}", True, RED)
    goat_text = font.render(f"Goat Wins: {goat_wins}", True, GREEN)
    current_player_text = font.render(f"Current Player: {'Goat' if current_player == 'goat' else 'Tiger'}", True, BLACK)
    
    screen.blit(tiger_text, (20, 20))
    screen.blit(goat_text, (WIDTH - 200, 20))
    screen.blit(current_player_text, (WIDTH // 2 - 100, 20))
    
    info_font = pygame.font.Font(None, 24)
    goats_to_place_text = info_font.render(f"Goats to place: {goats_to_place}", True, BLACK)
    captured_goats_text = info_font.render(f"Captured Goats: {captured_goats}", True, BLACK)
    move_count_text = info_font.render(f"Total Moves: {move_count}", True, BLACK)
    screen.blit(goats_to_place_text, (20, HEIGHT - 40))
    screen.blit(captured_goats_text, (WIDTH - 160, HEIGHT - 40))
    screen.blit(move_count_text, ((WIDTH // 2) - 50, HEIGHT - 40))

    pygame.display.flip()
    if game_over:
        if move_count >= 80:
            winner_text = font.render("Draw", True, BLACK)
        else:
            winner = "Tigers" if captured_goats >= 5 else "Goats"
            winner_text = font.render(f"{winner} Win!", True, BLACK)
        pygame.draw.rect(screen, WHITE, (WIDTH // 2 - 100, HEIGHT // 2 - 50, 200, 100))
        pygame.draw.rect(screen, BLACK, (WIDTH // 2 - 100, HEIGHT // 2 - 50, 200, 100), 2)
        screen.blit(winner_text, (WIDTH // 2 - 60, HEIGHT // 2 - 20))
        reset_text = info_font.render("Press 'R' to Restart", True, BLACK)
        screen.blit(reset_text, (WIDTH // 2 - 80, HEIGHT // 2 + 20))


def handle_scroll_event(event):
    global scroll_position
    
    if event.type == pygame.MOUSEWHEEL:
        if event.y > 0:  # Scrolling up
            scroll_position = max(0, scroll_position - 1)
        elif event.y < 0:  # Scrolling down
            max_scroll_position = max(0, len(moves) - max_visible_moves)
            scroll_position = min(scroll_position + 1, max_scroll_position)

        # Update UI immediately after scrolling
        draw_ui()

        # scroll_position = max(0, total_moves - max_visible_moves)+1


def handle_click_on_history(event):
    global scroll_position

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left-click
        pos = pygame.mouse.get_pos()
        if 20 <= pos[0] <= WIDTH - 60 and HEIGHT - 130 <= pos[1] <= HEIGHT - 40:  # Inside history box
            history_box_y = HEIGHT - 130
            
            max_scroll_position = max(0, len(moves) - max_visible_moves)
            
            # Calculate which row was clicked
            clicked_row = int((pos[1] - history_box_y) / SCROLL_STEP)
            if clicked_row < 0:
                clicked_row = 0
            
            # Adjust scroll position
            scroll_position = max(0, min(clicked_row * SCROLL_STEP, max_scroll_position))
            # Update UI immediately after clicking
            draw_ui()

def main():
    global current_player, selected_node, goats_to_place, captured_goats, game_over, tiger_wins, goat_wins, move_count, tiger_agent,scroll_position

    clock = pygame.time.Clock()
    tiger_agent = TigerAgent()
    tiger_agent.load_q_table()
    tiger_bot = TigerHeuristicBot(capture_connections, connections, capture_node)
    goat_bot = GoatHeuristicBot(connections, capture_connections, capture_node)

    while True:
        draw_board()
        draw_pieces()
        draw_ui()
        pygame.display.flip()
        
        for event in pygame.event.get():
            total_moves=len(moves)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            handle_scroll_event(event)
            handle_click_on_history(event)
            combobox_handling(event)

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                pos = pygame.mouse.get_pos()
                node = get_node_at_pos(pos)
                
                if node is not None:
                    if current_player == 'goat':
                        if goats_to_place > 0:
                            if node not in pieces['tigers'] and node not in pieces['goats']:
                                pieces['goats'].append(node)
                                goats_to_place -= 1
                                moves.append(f"Goat placed at node {node}")
                                move_count += 1
                                scroll_position = max(0, total_moves - max_visible_moves)+1
                                current_player = 'tiger'
                        else:
                            if selected_node is None and node in pieces['goats']:
                                selected_node = node
                            elif selected_node == node:
                                selected_node = None
                            elif selected_node is not None and is_valid_move(selected_node, node):
                                pieces['goats'].remove(selected_node)
                                pieces['goats'].append(node)
                                moves.append(f"Goat moved from {selected_node} to {node}")
                                selected_node = None
                                move_count += 1
                                scroll_position = max(0, total_moves - max_visible_moves)+1
                                current_player = 'tiger'

                    elif current_player == 'tiger' and game_mode == 'Goat vs Tiger':
                        if (len(pieces['tigers'])) < 3:
                            if node not in pieces['tigers'] and node not in pieces['goats']:
                                pieces['tigers'].append(node)
                                moves.append(f"Tiger placed at node {node}")
                                move_count += 1
                                scroll_position = max(0, total_moves - max_visible_moves)+1
                                current_player = 'goat'
                        elif selected_node is None and node in pieces['tigers']:
                            selected_node = node
                        elif selected_node == node:
                            selected_node = None
                        
                        elif selected_node is not None:
                            capture = is_capture_move(selected_node, node)
                            if capture:
                                pieces['tigers'].remove(selected_node)
                                pieces['tigers'].append(node)
                                pieces['goats'].remove(capture)
                                captured_goats += 1
                                moves.append(f"Tiger moved from {selected_node} to {node}, captured goat at {capture}")
                                selected_node = None
                                move_count += 1
                                scroll_position = max(0, total_moves - max_visible_moves)+1
                                current_player = 'goat'
                                if captured_goats >= 5:
                                    game_over = True
                                    tiger_wins += 1
                            elif is_valid_move(selected_node, node):
                                pieces['tigers'].remove(selected_node)
                                pieces['tigers'].append(node)
                                moves.append(f"Tiger moved from {selected_node} to {node}")
                                selected_node = None
                                scroll_position = max(0, total_moves - max_visible_moves)+1
                                move_count += 1
                                current_player = 'goat'

            elif event.type == pygame.KEYDOWN and game_over:
                if game_mode in ['Goat vs AI', 'Bot vs AI']:
                    tiger_agent.save_q_table()
                reset_game()

            # Handle AI and Bot moves outside the event loop
        if not game_over:
            if game_mode == 'Bot vs AI':
                if current_player == 'goat':
                    bot_move = goat_bot.make_move(pieces, goats_to_place)
                    selected_node, node = bot_move
                    if selected_node is None:  # Placement move
                        pieces['goats'].append(node)
                        goats_to_place -= 1
                        moves.append(f"Goat placed at node {node}")
                    else:  # Movement
                        pieces['goats'].remove(selected_node)
                        pieces['goats'].append(node)
                        moves.append(f"Goat moved from {selected_node} to {node}")
                        selected_node = None
                    move_count += 1
                    scroll_position = max(0, total_moves - max_visible_moves) + 1
                    current_player = 'tiger'
                else:  # Tiger's turn (AI)
                    if not are_tigers_trapped():
                        state = {'tigers': pieces['tigers'].copy(), 'goats': pieces['goats'].copy(), 'goats_to_place': goats_to_place}
                        action = tiger_agent.get_action(state)
                        selected_node, node = action
                        if selected_node is not None:
                            capture = is_capture_move(selected_node, node)
                            pieces['tigers'].remove(selected_node)
                            pieces['tigers'].append(node)
                            if capture:
                                moves.append(f"Tiger moved from {selected_node} to {node}, captured goat at {capture}")
                                pieces['goats'].remove(capture)
                                selected_node = None
                                captured_goats += 1
                                reward = 10
                                if captured_goats >= 5:
                                    game_over = True
                                    tiger_wins += 1
                            else:
                                moves.append(f"Tiger moved from {selected_node} to {node}")
                                selected_node = None
                                reward = -1
                        else:
                            reward = 0.25
                            pieces['tigers'].append(node)
                            moves.append(f"Tiger placed at node {node}")

                        next_state = {'tigers': pieces['tigers'].copy(), 'goats': pieces['goats'].copy(), 'goats_to_place': goats_to_place}
                        tiger_agent.update_q_table(state, action, next_state, reward)
                        move_count += 1
                        scroll_position = max(0, total_moves - max_visible_moves) + 1
                        current_player = 'goat'
            else:
                if not game_over and current_player == 'tiger':
                    if game_mode == 'Goat vs Bot':
                        if not are_tigers_trapped():
                            if (len(pieces['tigers'])) < 3:
                                    node=tiger_bot.get_placement_move(pieces)
                                    pieces['tigers'].append(node)
                                    moves.append(f"Tiger placed at node {node}")
                                    current_player = 'goat'
                                    scroll_position = max(0, total_moves - max_visible_moves)+1
                                    move_count += 1
                            else:
                                bot_move = tiger_bot.get_best_move(pieces)
                                if bot_move:
                                    selected_node, node = bot_move
                                    capture = is_capture_move(selected_node, node)
                                    pieces['tigers'].remove(selected_node)
                                    pieces['tigers'].append(node)
                                    if capture:
                                        moves.append(f"Tiger moved from {selected_node} to {node}, captured goat at {capture}")
                                        pieces['goats'].remove(capture)
                                        captured_goats += 1
                                        if captured_goats >= 5:
                                            game_over = True
                                            tiger_wins += 1
                                    else:
                                        moves.append(f"Tiger moved from {selected_node} to {node}")
                                    selected_node = None
                                    current_player = 'goat'
                                    scroll_position = max(0, total_moves - max_visible_moves)+1
                                move_count += 1
                    elif game_mode == 'Goat vs AI':
                        if not are_tigers_trapped():
                            state = {'tigers': pieces['tigers'].copy(), 'goats': pieces['goats'].copy(), 'goats_to_place': goats_to_place}
                            action = tiger_agent.get_action(state)
                            selected_node, node = action
                            if selected_node!=None:
                                capture = is_capture_move(selected_node, node)
                                pieces['tigers'].remove(selected_node)
                                pieces['tigers'].append(node)
                                if capture:
                                    moves.append(f"Tiger moved from {selected_node} to {node}, captured goat at {capture}")
                                    pieces['goats'].remove(capture)
                                    selected_node = None
                                    captured_goats += 1
                                    reward = 10
                                    if captured_goats >= 5:
                                        game_over = True
                                        tiger_wins += 1
                                else:
                                    moves.append(f"Tiger moved from {selected_node} to {node}")
                                    selected_node = None
                                    reward = -1
                            else:
                                reward = 0.25
                                pieces['tigers'].append(node)

                            next_state = {'tigers': pieces['tigers'].copy(), 'goats': pieces['goats'].copy(), 'goats_to_place': goats_to_place}
                            tiger_agent.update_q_table(state, action, next_state, reward)
                            current_player = 'goat'
                            scroll_position = max(0, total_moves - max_visible_moves)+1
                            move_count += 1
                

            if not game_over:
                if are_tigers_trapped():
                    game_over = True
                    goat_wins += 1
                elif move_count >= 80:
                    game_over = True

        clock.tick(FPS)

if __name__ == "__main__":
    main()


