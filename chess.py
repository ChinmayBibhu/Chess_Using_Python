import matplotlib.pyplot as plt

class Chessboard:
    def __init__(self, size=8):
        self.game_over= False
        self.size = size
        self.fig, self.ax = plt.subplots()
        self.pieces = {}
        self.selected_piece = None
        self.game_state_history = []
        self.fifty_move_counter = 0
        self.draw_reason = None
        self.move_dots = []
        self.current_turn = 'white' 
        self.white_pieces = {'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙'}
        self.black_pieces = {'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟'}
        self.white_symbols = {v: ('white', k) for k, v in self.white_pieces.items()}
        self.black_symbols = {v: ('black', k) for k, v in self.black_pieces.items()}
        
        self.moved_pieces = set()
        self.en_passant_target = None
        self.draw_board()
        self.piece_setup()
        self.setup_events()
    
    def setup_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def handle_movement(self, start_pos, end_pos):
        start_col, start_row = start_pos
        end_col, end_row = end_pos
        if (end_col, end_row) in self.available_moves(*start_pos):
            piece = self.pieces.pop(start_pos)
            
            # Track moved pieces for castling
            if piece in [self.white_pieces['K'], self.black_pieces['K']] or \
               piece in [self.white_pieces['R'], self.black_pieces['R']]:
                self.moved_pieces.add((start_col, start_row))
            
            # Handle en passant capture
            if piece in [self.white_pieces['P'], self.black_pieces['P']]:
                # Check for en passant
                if end_pos == self.en_passant_target:
                    captured_row = end_row + 1 if self.current_turn == 'white' else end_row - 1
                    del self.pieces[(end_col, captured_row)]
                
                # Set en passant target for double pawn moves
                if abs(start_row - end_row) == 2:
                    self.en_passant_target = (end_col, (start_row + end_row) // 2)
                else:
                    self.en_passant_target = None
            else:
                self.en_passant_target = None

            # Handle castling
            if piece in [self.white_pieces['K'], self.black_pieces['K']] and abs(start_col - end_col) == 2:
                # Move the rook
                rook_col = 7 if end_col == 6 else 0
                rook_new_col = 5 if end_col == 6 else 3
                rook_pos = (rook_col, start_row)
                rook = self.pieces.pop(rook_pos)
                self.pieces[(rook_new_col, start_row)] = rook
                self.moved_pieces.add(rook_pos)

            self.pieces[end_pos] = piece


            # Track move for 50-move rule
            moved_piece = piece
            captured = end_pos in self.pieces
            is_pawn = moved_piece in [self.white_pieces['P'], self.black_pieces['P']]
    
            if is_pawn or captured:
                self.fifty_move_counter = 0
            else:
                self.fifty_move_counter += 1
    
    # Track game state for threefold repetition
            self.game_state_history.append((
                frozenset(self.pieces.items()),
                self.current_turn,
                frozenset(self.moved_pieces),
                self.en_passant_target
    ))

            return True
        return False
    
    def on_click(self, event):
        if self.game_over or not event.inaxes:
            return
    
        col = int(event.xdata)
        row = int(event.ydata)
    
        if 0 <= col < self.size and 0 <= row < self.size:
        # Clear move dots
            for dot in self.move_dots:
                dot.remove()
            self.move_dots = []
        
            if self.selected_piece:
                if self.handle_movement(self.selected_piece, (col, row)):
                    opponent_color = 'black' if self.current_turn == 'white' else 'white'
                
                # Check draw conditions first
                    draw_detected = False
                    if self.insufficient_material():
                        self.draw_reason = "Draw: Insufficient Material"
                        draw_detected = True
                    elif self.threefold_repetition():
                        self.draw_reason = "Draw: Threefold Repetition"
                        draw_detected = True
                    elif self.check_fifty_move_rule():
                        self.draw_reason = "Draw: Fifty-Move Rule"
                        draw_detected = True
                    elif self.is_stalemate(opponent_color):
                        self.draw_reason = "Draw: Stalemate"
                        draw_detected = True
                
                # Then check for checkmate
                    if self.is_checkmate(opponent_color):
                        self.draw_reason = None  # Clear draw reason if checkmate exists
                        self.game_over = True
                
                # Update game state if any condition met
                    if draw_detected:
                        self.game_over = True
                
                    if self.game_over:
                        self.redraw_board()
                        return  # Prevent further moves
                
                # Switch turns if game continues
                    self.current_turn = 'black' if self.current_turn == 'white' else 'white'
                    self.redraw_board()
            
                self.selected_piece = None
            elif (col, row) in self.pieces:
                piece = self.pieces[(col, row)]
                piece_color = 'white' if piece in self.white_symbols else 'black'
                if piece_color == self.current_turn:
                    self.selected_piece = (col, row)
                    moves = self.available_moves(col, row)
                    for (c, r) in moves:
                        dot = self.ax.scatter(c + 0.5, r + 0.5, color='red', s=100, zorder=3)
                        self.move_dots.append(dot)
                plt.draw()

    def redraw_board(self):
        """Redraw the entire board"""
        self.ax.clear()
        self.draw_board()
        self.draw_pieces()
    
    # Single game-over message block
        if self.game_over:
            if hasattr(self, 'draw_reason') and self.draw_reason:
                text = self.draw_reason
                font_size = 24
            else:
                text = f"CHECKMATE!\n{self.current_turn.capitalize()} WINS!"
                font_size = 30
        
            self.ax.text(self.size/2, self.size/2, 
                   text, 
                   ha='center', va='center', 
                   fontsize=font_size, color='red',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Highlight king in check (single instance)
        current_color = self.current_turn
        king_pos = self.find_king(current_color)
        if king_pos and self.is_in_check(king_pos, current_color, self.pieces):
            self.highlight_king(king_pos)
    
        plt.draw()

    def draw_pieces(self):
        """Draw pieces with correct team colors"""
        for (col, row), symbol in self.pieces.items():
        # Determine color based on piece team
            if symbol in self.white_symbols:
                text_color = 'white'  # White pieces
            elif symbol in self.black_symbols:
                text_color = 'black'  # Black pieces
            else:
                text_color = 'red'    # Error color (should never happen)
            
            self.ax.text(
                col + 0.5, row + 0.5, symbol,
                ha='center', va='center',
                fontsize=20, color=text_color
            )
    
    def draw_board(self):
        self.ax.clear()
        for row in range(self.size):
            for col in range(self.size):
                color = '#f0d9b5' if (row + col) % 2 == 0 else '#b58863'
                self.ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color))
        
        self.ax.set_xticks([i + 0.5 for i in range(self.size)])
        self.ax.set_yticks([i + 0.5 for i in range(self.size)])
        self.ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], fontsize=12)
        self.ax.set_yticklabels([str(i) for i in range(1, self.size + 1)], fontsize=12)
        self.ax.tick_params(axis='both', which='both', length=0)
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_aspect('equal')
        self.ax.add_patch(plt.Rectangle((0, 0), self.size, self.size, 
                                      linewidth=2, edgecolor='black', facecolor='none'))

    def piece_setup(self):
        # White pieces (fixed color parameter)
        for col in range(self.size):
            self._add_piece(col, 6, self.white_pieces['P'], 'white')  # Changed to 'white'
        back_row = [self.white_pieces['R'], self.white_pieces['N'], 
                   self.white_pieces['B'], self.white_pieces['Q'],
                   self.white_pieces['K'], self.white_pieces['B'],
                   self.white_pieces['N'], self.white_pieces['R']]
        for col, piece in enumerate(back_row):
            self._add_piece(col, 7, piece, 'white')  # Changed to 'white'

        # Black pieces (fixed color parameter)
        for col in range(self.size):
            self._add_piece(col, 1, self.black_pieces['P'], 'black')  # Changed to 'black'
        back_row = [self.black_pieces['R'], self.black_pieces['N'],
                   self.black_pieces['B'], self.black_pieces['Q'],
                   self.black_pieces['K'], self.black_pieces['B'],
                   self.black_pieces['N'], self.black_pieces['R']]
        for col, piece in enumerate(back_row):
            self._add_piece(col, 0, piece, 'black')  # Changed to 'black'

    def _add_piece(self, col, row, symbol, color):
        self.ax.text(col + 0.5, row + 0.5, symbol, 
                   ha='center', va='center', 
                   fontsize=20, color=color)
        self.pieces[(col, row)] = symbol

    def available_moves(self, col, row):
        piece = self.pieces.get((col, row))
        if not piece:
            return []

        if piece in self.white_symbols:
            color, p_type = self.white_symbols[piece]
        elif piece in self.black_symbols:
            color, p_type = self.black_symbols[piece]
        else:
            return []

        moves = []
        if p_type == 'P':
            moves = self._pawn_moves(col, row, color)
        elif p_type == 'N':
            moves = self._knight_moves(col, row, color)
        elif p_type == 'B':
            moves = self._bishop_moves(col, row, color)
        elif p_type == 'R':
            moves = self._rook_moves(col, row, color)
        elif p_type == 'Q':
            moves = self._queen_moves(col, row, color)
        elif p_type == 'K':
            moves = self._king_moves(col, row, color)
    
        valid_moves = self._filter_moves(moves, color)
    
    # Filter moves that would leave king in check
        king_pos = self.find_king(color)
        final_moves = []
        for move in valid_moves:
            if not self.move_exposes_king((col, row), move, color, king_pos):
                final_moves.append(move)
        return final_moves
    
    def find_king(self, color):
        king = self.white_pieces['K'] if color == 'white' else self.black_pieces['K']
        for pos, piece in self.pieces.items():
            if piece == king:
                return pos
        return None

    def highlight_king(self, position):
        """Highlight the king's square in red if in check"""
        col, row = position
        self.ax.add_patch(plt.Rectangle(
            (col, row), 1, 1,
            color='red', alpha=0.3, zorder=1
        ))

    def move_exposes_king(self, start_pos, end_pos, color, king_pos):
    # Create temporary board state
        temp_pieces = self.pieces.copy()
        moving_piece = temp_pieces[start_pos]
    
    # Update king position if moving the king
        if moving_piece == self.white_pieces['K'] or moving_piece == self.black_pieces['K']:
            king_pos = end_pos
    
    # Apply the move
        del temp_pieces[start_pos]
        temp_pieces[end_pos] = moving_piece
    
    # Check if king is in check
        return self.is_in_check(king_pos, color, temp_pieces)

    def is_in_check(self, king_pos, color, pieces):
        opponent_color = 'black' if color == 'white' else 'white'
    
        for pos, symbol in pieces.items():
            if (color == 'white' and symbol in self.black_symbols) or \
                (color == 'black' and symbol in self.white_symbols):
        
                if symbol in self.white_symbols:
                    _, p_type = self.white_symbols[symbol]
                else:
                    _, p_type = self.black_symbols[symbol]
        
        # Generate moves for this opponent's piece
                moves = []
                if p_type == 'P':
                    moves = self._pawn_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'N':
                    moves = self._knight_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'B':
                    moves = self._bishop_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'R':
                    moves = self._rook_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'Q':
                    moves = self._queen_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'K':
                    moves = self._king_moves(pos[0], pos[1], opponent_color, pieces, check_castling=False)
        
                if king_pos in moves:
                    return True
        return False

    def _pawn_moves(self, col, row, color, pieces=None):
        pieces = self.pieces if pieces is None else pieces
        moves = []
        direction = -1 if color == 'white' else 1
        start_row = 6 if color == 'white' else 1

        # Forward moves
        if (col, row + direction) not in pieces:
            moves.append((col, row + direction))
            if row == start_row and (col, row + 2*direction) not in pieces:
                moves.append((col, row + 2*direction))

        # Captures
        for dx in [-1, 1]:
            new_col = col + dx
            new_row = row + direction
            if 0 <= new_col < self.size and 0 <= new_row < self.size:
                # Normal capture
                if (new_col, new_row) in pieces:
                    if color == 'white' and pieces[(new_col, new_row)] in self.black_symbols:
                        moves.append((new_col, new_row))
                    elif color == 'black' and pieces[(new_col, new_row)] in self.white_symbols:
                        moves.append((new_col, new_row))
                # En passant
                elif (new_col, new_row) == self.en_passant_target:
                    moves.append((new_col, new_row))
        
        return moves

    def _knight_moves(self, col, row, color, pieces=None):
        pieces = self.pieces if pieces is None else pieces
        moves = []
        knight_jumps = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for dx, dy in knight_jumps:
            new_col = col + dx
            new_row = row + dy
            if 0 <= new_col < self.size and 0 <= new_row < self.size:
                occupied_piece = pieces.get((new_col, new_row))
            # Check using pieces instead of self.pieces
                if occupied_piece is None or \
                (color == 'white' and occupied_piece in self.black_symbols) or \
                (color == 'black' and occupied_piece in self.white_symbols):
                    moves.append((new_col, new_row))
        return moves

    def _bishop_moves(self, col, row, color,pieces=None):
        pieces = self.pieces if pieces is None else pieces
        return self._slide_moves(col, row, color, [(1,1), (1,-1), (-1,1), (-1,-1)], pieces)

    def _rook_moves(self, col, row, color, pieces=None):
        pieces = self.pieces if pieces is None else pieces
        return self._slide_moves(col, row, color, [(1,0), (-1,0), (0,1), (0,-1)], pieces)

    def _queen_moves(self, col, row, color, pieces=None):
        pieces = self.pieces if pieces is None else pieces
        return self._slide_moves(col, row, color, 
                               [(1,0), (-1,0), (0,1), (0,-1),
                                (1,1), (1,-1), (-1,1), (-1,-1)], pieces)

    def is_attacked(self, position, color, pieces):
        opponent_color = 'black' if color == 'white' else 'white'
        for pos, symbol in pieces.items():
            if (color == 'white' and symbol in self.black_symbols) or \
               (color == 'black' and symbol in self.white_symbols):
                
                if symbol in self.white_symbols:
                    _, p_type = self.white_symbols[symbol]
                else:
                    _, p_type = self.black_symbols[symbol]
                
                moves = []
                if p_type == 'P':
                    moves = self._pawn_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'N':
                    moves = self._knight_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'B':
                    moves = self._bishop_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'R':
                    moves = self._rook_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'Q':
                    moves = self._queen_moves(pos[0], pos[1], opponent_color, pieces)
                elif p_type == 'K':
                    moves = self._king_moves(pos[0], pos[1], opponent_color, pieces, check_castling=False)
                
                if position in moves:
                    return True
        return False

    def _king_moves(self, col, row, color, pieces=None, check_castling=True):  # Modified line
        pieces = self.pieces if pieces is None else pieces
        moves = []
        king = self.white_pieces['K'] if color == 'white' else self.black_pieces['K']
    
    # Normal king moves
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_col = col + dx
                new_row = row + dy
                if 0 <= new_col < self.size and 0 <= new_row < self.size:
                    if (new_col, new_row) not in pieces or \
                        (color == 'white' and pieces[(new_col, new_row)] in self.black_symbols) or \
                        (color == 'black' and pieces[(new_col, new_row)] in self.white_symbols):
                        moves.append((new_col, new_row))
    
    # Castling (only if check_castling is True)
        if check_castling:  # Added condition
            if (col, row) not in self.moved_pieces and not self.is_in_check((col, row), color, pieces):
            # Queenside castling
                if (0, row) not in self.moved_pieces:
                    if all((c, row) not in pieces for c in [1, 2, 3]):
                        if not any(self.is_attacked((c, row), color, pieces) for c in [2, 3, 4]):
                            moves.append((col-2, row))
            
            # Kingside castling
                if (7, row) not in self.moved_pieces:
                    if all((c, row) not in pieces for c in [5, 6]):
                        if not any(self.is_attacked((c, row), color, pieces) for c in [4, 5, 6]):
                         moves.append((col+2, row))
        
        return moves

    def _slide_moves(self, col, row, color, directions, pieces=None):
        pieces = self.pieces if pieces is None else pieces
        moves = []
        for dx, dy in directions:
            step = 1
            while True:
                new_col = col + dx * step
                new_row = row + dy * step
                if not (0 <= new_col < self.size and 0 <= new_row < self.size):
                    break
            
            # Check if square exists in pieces using .get()
                occupied_piece = pieces.get((new_col, new_row))
                if occupied_piece is not None:
                    if (color == 'white' and occupied_piece in self.black_symbols) or \
                       (color == 'black' and occupied_piece in self.white_symbols):
                        moves.append((new_col, new_row))
                    break
                moves.append((new_col, new_row))
                step += 1
        return moves

    def is_stalemate(self, color):
    # Check if player has no legal moves and is not in check
        if self.is_in_check(self.find_king(color), color, self.pieces):
            return False
    
        for pos, piece in self.pieces.items():
            piece_color = 'white' if piece in self.white_symbols else 'black'
            if piece_color == color and len(self.available_moves(*pos)) > 0:
                return False
        return True

    def insufficient_material(self):
    # Count remaining pieces
        white_pieces = []
        black_pieces = []
        for piece in self.pieces.values():
            if piece in self.white_symbols:
                white_pieces.append(self.white_symbols[piece][1])
            else:
                black_pieces.append(self.black_symbols[piece][1])
    
    # King vs King
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True
    
    # King + minor vs King
    def is_insufficient(pieces):
            return len(pieces) == 2 and any(p in ['B', 'N'] for p in pieces)
    
            if (is_insufficient(white_pieces) and len(black_pieces) == 1 or
                is_insufficient(black_pieces) and len(white_pieces) == 1):
                return True
    
    # King + Bishop vs King + Bishop (same color)
            if (len(white_pieces) == 2 and white_pieces.count('B') == 1 and
                len(black_pieces) == 2 and black_pieces.count('B') == 1):
                white_bishop = [p for p in white_pieces if p == 'B'][0]
                black_bishop = [p for p in black_pieces if p == 'B'][0]
        # Check bishop colors (assuming even+even or odd+odd squares)
        # This would require tracking bishop positions - omitted for brevity
                return True  # Simplified check
    
            return False

    def threefold_repetition(self):
    # Create state signature
        state = (
            frozenset(self.pieces.items()),
            self.current_turn,
            frozenset(self.moved_pieces),
            self.en_passant_target
        )
    # Count occurrences
        count = sum(1 for s in self.game_state_history if s == state)
        return count >= 2  # Third occurrence

    def check_fifty_move_rule(self):
        return self.fifty_move_counter >= 100  # 50 full moves

    def is_checkmate(self, color):
        king_pos = self.find_king(color)
        if not king_pos:
            return False  # Shouldn't happen, but safety check
    
    # Condition 1: King is in check
        if not self.is_in_check(king_pos, color, self.pieces):
            return False
    
    # Condition 2: King has no valid moves
        king_moves = self._king_moves(king_pos[0], king_pos[1], color)
        for move in king_moves:
            temp_pieces = self.pieces.copy()
        # Simulate moving the king
            king_piece = temp_pieces.pop(king_pos)
            temp_pieces[move] = king_piece
            if not self.is_in_check(move, color, temp_pieces):
                return False  # King can escape
    
    # Condition 3: No blocking or capturing moves
        attackers = self.get_attackers(king_pos, color)
        if len(attackers) > 1:
            return True  # Double check; no blocks possible
    
    # Single attacker: Check if any piece can capture or block
        attacker_pos = attackers[0]
        attacker_piece = self.pieces[attacker_pos]
        attacker_type = self.black_symbols[attacker_piece][1] if color == 'white' else self.white_symbols[attacker_piece][1]
    
    # Path squares between attacker and king (for blocking)
        path = self.get_attack_path(king_pos, attacker_pos, attacker_type)
    
    # Check if any piece can capture the attacker
        for pos, piece in self.pieces.items():
            piece_color, piece_type = (self.white_symbols[piece] if piece in self.white_symbols 
                                    else self.black_symbols[piece])
            if piece_color != color:
                continue
            if pos == king_pos:
                continue  # Skip the king
            moves = self.available_moves(pos[0], pos[1])
            if attacker_pos in moves:
                return False  # Attacker can be captured
    
    # Check if any piece can block the path
        if attacker_type in ['B', 'R', 'Q']:
            for square in path:
                for pos, piece in self.pieces.items():
                    piece_color, _ = (self.white_symbols[piece] if piece in self.white_symbols 
                                   else self.black_symbols[piece])
                    if piece_color != color:
                        continue
                    if pos == king_pos:
                        continue
                    moves = self.available_moves(pos[0], pos[1])
                    if square in moves:
                        return False  # Path can be blocked
    
        return True  # All conditions met for checkmate

    def _filter_moves(self, moves, color):
        valid = []
        for (c, r) in moves:
            if (c, r) in self.pieces:
                piece = self.pieces[(c, r)]
                if color == 'white' and piece in self.white_symbols:
                    continue
                if color == 'black' and piece in self.black_symbols:
                    continue
            valid.append((c, r))
        return valid
    
    def get_attackers(self, king_pos, color):
        attackers = []
        opponent_color = 'black' if color == 'white' else 'white'
        for pos, symbol in self.pieces.items():
            if (color == 'white' and symbol in self.black_symbols) or \
                (color == 'black' and symbol in self.white_symbols):
            # Check if this piece can attack the king
                moves = self.available_moves(pos[0], pos[1])
                if king_pos in moves:
                    attackers.append(pos)
        return attackers

    def get_attack_path(self, king_pos, attacker_pos, attacker_type):
        if attacker_type not in ['B', 'R', 'Q']:
            return []  # No path for non-sliding pieces
    
        path = []
        dx = attacker_pos[0] - king_pos[0]
        dy = attacker_pos[1] - king_pos[1]
    
        if dx == 0:  # Vertical
            step = 1 if attacker_pos[1] < king_pos[1] else -1
            for y in range(attacker_pos[1] + step, king_pos[1], step):
                path.append((attacker_pos[0], y))
        elif dy == 0:  # Horizontal
            step = 1 if attacker_pos[0] < king_pos[0] else -1
            for x in range(attacker_pos[0] + step, king_pos[0], step):
                path.append((x, attacker_pos[1]))
        else:  # Diagonal
            x_step = 1 if attacker_pos[0] < king_pos[0] else -1
            y_step = 1 if attacker_pos[1] < king_pos[1] else -1
            distance = abs(attacker_pos[0] - king_pos[0])
            for i in range(1, distance):
                path.append((attacker_pos[0] + i * x_step, attacker_pos[1] + i * y_step))
        return path
    
    def add_click_event(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    

    def show(self):
        plt.show()

# Create and display the chessboard
chessboard = Chessboard()
chessboard.show()