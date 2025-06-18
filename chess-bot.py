import pyautogui
import time
import cv2
import numpy as np
import os
import subprocess

class ChessBot:
    def __init__(self):
        # Timer coordinates for turn detection
        self.TIMER_X, self.TIMER_Y = 635, 620
        
        # Board capture coordinates
        self.board_x1, self.board_y1 = 561, 262
        self.board_x2, self.board_y2 = 805, 506
        
        # Known colors for turn detection
        self.WHITE_TURN_RGB = (255, 255, 255)
        self.GREY_TURN_RGB = (135, 135, 135)
        
        # Paths
        self.base_path = r'C:\Users\rahul\Desktop\New folder'
        self.board_path = os.path.join(self.base_path, 'board.png')
        self.stockfish_path = 'stockfish.exe'
        
        # Player color (will be detected automatically)
        self.player_color = None
        
        # Piece values for evaluation
        self.piece_values = {
            'p': 1, 'P': 1,    # Pawn
            'n': 3, 'N': 3,    # Knight
            'b': 3, 'B': 3,    # Bishop
            'r': 5, 'R': 5,    # Rook
            'q': 9, 'Q': 9,    # Queen
            'k': 100, 'K': 100 # King (very high value)
        }
        
        # Square to coordinate mapping
        self.square_to_coord = {
            'a1': (575, 490), 'a2': (575, 460), 'a3': (575, 430), 'a4': (575, 400),
            'a5': (575, 370), 'a6': (575, 338), 'a7': (575, 307), 'a8': (575, 275),
            'b1': (605, 490), 'b2': (605, 460), 'b3': (605, 430), 'b4': (605, 400),
            'b5': (605, 370), 'b6': (605, 338), 'b7': (605, 307), 'b8': (605, 275),
            'c1': (635, 490), 'c2': (635, 460), 'c3': (635, 430), 'c4': (635, 400),
            'c5': (635, 370), 'c6': (635, 338), 'c7': (635, 307), 'c8': (635, 275),
            'd1': (666, 490), 'd2': (666, 460), 'd3': (666, 430), 'd4': (666, 400),
            'd5': (666, 370), 'd6': (666, 338), 'd7': (666, 307), 'd8': (666, 275),
            'e1': (697, 490), 'e2': (697, 460), 'e3': (697, 430), 'e4': (697, 400),
            'e5': (697, 370), 'e6': (697, 338), 'e7': (697, 307), 'e8': (697, 275),
            'f1': (727, 490), 'f2': (727, 460), 'f3': (727, 430), 'f4': (727, 400),
            'f5': (727, 370), 'f6': (727, 338), 'f7': (727, 307), 'f8': (727, 275),
            'g1': (757, 490), 'g2': (757, 460), 'g3': (757, 430), 'g4': (757, 400),
            'g5': (757, 370), 'g6': (757, 338), 'g7': (757, 307), 'g8': (757, 275),
            'h1': (789, 490), 'h2': (789, 460), 'h3': (789, 430), 'h4': (789, 400),
            'h5': (789, 370), 'h6': (789, 338), 'h7': (789, 307), 'h8': (789, 275),
        }
        
        # Load templates
        self.load_templates()
    
    def is_similar_color(self, rgb1, rgb2, tolerance=20):
        """Check if two RGB colors are similar within tolerance"""
        return all(abs(a - b) <= tolerance for a, b in zip(rgb1, rgb2))
    
    def detect_turn(self):
        """Detect whose turn it is based on timer color"""
        pixel_color = pyautogui.screenshot().getpixel((self.TIMER_X, self.TIMER_Y))
        
        if self.is_similar_color(pixel_color, self.WHITE_TURN_RGB):
            return "your"
        elif self.is_similar_color(pixel_color, self.GREY_TURN_RGB):
            return "opponent"
        else:
            return "unknown"
    
    def detect_player_color(self):
        """Detect if player is playing as White or Black by checking piece positions"""
        if not self.capture_board():
            return None
            
        try:
            board = cv2.imread(self.board_path)
            if board is None:
                return None
                
            board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
            height, width = board_gray.shape
            cell_h, cell_w = height // 8, width // 8
            
            # Check bottom-left corner (should be a1 for White, h8 for Black)
            # If there's a white rook at bottom-left, player is White
            # If there's a black rook at bottom-left, player is Black
            
            bottom_left_cell = board_gray[7*cell_h:8*cell_h, 0:cell_w]
            
            # Check for white rook template
            if 'wr_w' in self.templates and self.templates['wr_w'] is not None:
                res = cv2.matchTemplate(bottom_left_cell, self.templates['wr_w'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    print("🟢 Detected: Playing as WHITE")
                    return 'white'
            
            if 'wr_b' in self.templates and self.templates['wr_b'] is not None:
                res = cv2.matchTemplate(bottom_left_cell, self.templates['wr_b'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    print("🟢 Detected: Playing as WHITE")
                    return 'white'
            
            # Check for black rook template
            if 'br_w' in self.templates and self.templates['br_w'] is not None:
                res = cv2.matchTemplate(bottom_left_cell, self.templates['br_w'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    print("🔴 Detected: Playing as BLACK")
                    return 'black'
                    
            if 'br_b' in self.templates and self.templates['br_b'] is not None:
                res = cv2.matchTemplate(bottom_left_cell, self.templates['br_b'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    print("🔴 Detected: Playing as BLACK")
                    return 'black'
            
            # Fallback: check for any white pieces in bottom two rows
            white_pieces_bottom = 0
            black_pieces_bottom = 0
            
            for row in range(6, 8):  # Bottom 2 rows
                for col in range(8):
                    y1, y2 = row * cell_h, (row + 1) * cell_h
                    x1, x2 = col * cell_w, (col + 1) * cell_w
                    cell = board_gray[y1:y2, x1:x2]
                    
                    for name, template in self.templates.items():
                        if template is None:
                            continue
                        res = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        
                        if max_val > 0.6:
                            if name.startswith('w'):  # White piece
                                white_pieces_bottom += 1
                            elif name.startswith('b'):  # Black piece
                                black_pieces_bottom += 1
            
            if white_pieces_bottom > black_pieces_bottom:
                print("🟢 Detected: Playing as WHITE (based on piece distribution)")
                return 'white'
            elif black_pieces_bottom > white_pieces_bottom:
                print("🔴 Detected: Playing as BLACK (based on piece distribution)")
                return 'black'
            
            print("⚠️ Could not detect player color, defaulting to WHITE")
            return 'white'
            
        except Exception as e:
            print(f"❌ Error detecting player color: {e}")
            return 'white'  # Default to white
    
    def capture_board(self):
        """Capture the chess board and save as image"""
        try:
            width = self.board_x2 - self.board_x1
            height = self.board_y2 - self.board_y1
            screenshot = pyautogui.screenshot(region=(self.board_x1, self.board_y1, width, height))
            screenshot.save(self.board_path)
            print("✅ Board screenshot captured successfully")
            return True
        except Exception as e:
            print(f"❌ Error capturing board: {e}")
            return False
    
    def load_templates(self):
        """Load all piece templates"""
        self.templates = {}
        template_names = [
            'wp_w', 'wp_b', 'bp_w', 'bp_b', 'wr_w', 'wr_b', 'br_w', 'br_b',
            'wn_w', 'wn_b', 'bn_w', 'bn_b', 'wb_w', 'wb_b', 'bb_w', 'bb_b',
            'wq_w', 'wq_b', 'bq_w', 'bq_b', 'wk_w', 'wk_b', 'bk_w', 'bk_b'
        ]
        
        for name in template_names:
            template_path = os.path.join(self.base_path, f'{name}.png')
            if os.path.exists(template_path):
                self.templates[name] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"⚠️ Template {name}.png not found")
                self.templates[name] = None
    
    def generate_fen(self):
        """Generate FEN string from board image"""
        try:
            # Load board image
            board = cv2.imread(self.board_path)
            if board is None:
                print("❌ Could not load board image")
                return None
                
            board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
            height, width = board_gray.shape
            cell_h, cell_w = height // 8, width // 8
            
            threshold = 0.65
            matched_positions = []
            
            # Board mapping depends on player color
            if self.player_color == 'white':
                # White at bottom: top-left is a8, bottom-right is h1
                files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
            else:
                # Black at bottom: top-left is h1, bottom-right is a8
                files = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
                ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
            
            for row in range(8):
                for col in range(8):
                    y1, y2 = row * cell_h, (row + 1) * cell_h
                    x1, x2 = col * cell_w, (col + 1) * cell_w
                    cell = board_gray[y1:y2, x1:x2]
                    best_match = None
                    best_score = 0
                    
                    for name, template in self.templates.items():
                        if template is None:
                            continue
                        res = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        
                        if max_val > best_score and max_val >= threshold:
                            best_score = max_val
                            best_match = name
                    
                    if best_match:
                        color = best_match[0]  # w or b
                        piece = best_match[1]  # p, r, n, b, q, k
                        file = files[col]
                        rank = ranks[row]
                        pos_str = f"{color}{piece}{file}{rank}"
                        matched_positions.append(pos_str)
            
            # Generate FEN from positions
            fen = self.positions_to_fen(matched_positions)
            print(f"✅ Generated FEN: {fen}")
            return fen
            
        except Exception as e:
            print(f"❌ Error generating FEN: {e}")
            return None
    
    def positions_to_fen(self, positions):
        """Convert piece positions to FEN string"""
        board = [['' for _ in range(8)] for _ in range(8)]
        piece_map = {'p': 'p', 'r': 'r', 'n': 'n', 'b': 'b', 'q': 'q', 'k': 'k'}
        
        for item in positions:
            if len(item) >= 4:
                color, piece, file, rank = item[0], item[1], item[2], item[3]
                row = 8 - int(rank)
                col = ord(file) - ord('a')
                symbol = piece_map[piece].upper() if color == 'w' else piece_map[piece]
                board[row][col] = symbol
        
        fen_rows = []
        for row in board:
            fen_row = ''
            empty = 0
            for cell in row:
                if cell == '':
                    empty += 1
                else:
                    if empty:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += cell
            if empty:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        
        # Set the active color based on player color and turn
        active_color = 'w' if self.player_color == 'white' else 'b'
        
        return '/'.join(fen_rows) + f' {active_color} - - 0 1'

    def get_multiple_best_moves(self, fen, depth=18, num_moves=2):
        """Get multiple best moves from Stockfish with their evaluations"""
        try:
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Set MultiPV to get multiple best moves
            commands = [
                f'setoption name MultiPV value {num_moves}',
                f'position fen {fen}',
                f'go depth {depth}'
            ]
            
            for command in commands:
                process.stdin.write(command + '\n')
            process.stdin.flush()
            
            moves_with_eval = []
            
            while True:
                output = process.stdout.readline()
                if not output:
                    break
                    
                # Parse PV lines for moves and evaluations
                if 'info depth' in output and 'pv' in output and f'depth {depth}' in output:
                    parts = output.split()
                    try:
                        # Find evaluation
                        eval_score = 0
                        if 'cp' in parts:
                            cp_index = parts.index('cp')
                            if cp_index + 1 < len(parts):
                                eval_score = int(parts[cp_index + 1])
                        elif 'mate' in parts:
                            mate_index = parts.index('mate')
                            if mate_index + 1 < len(parts):
                                mate_in = int(parts[mate_index + 1])
                                eval_score = 10000 if mate_in > 0 else -10000
                        
                        # Find the move (first move in PV)
                        if 'pv' in parts:
                            pv_index = parts.index('pv')
                            if pv_index + 1 < len(parts):
                                move = parts[pv_index + 1]
                                moves_with_eval.append((move, eval_score))
                    except (ValueError, IndexError):
                        continue
                        
                if 'bestmove' in output:
                    break
            
            process.stdin.write('quit\n')
            process.stdin.flush()
            process.terminate()
            
            # Sort by evaluation (best first)
            moves_with_eval.sort(key=lambda x: x[1], reverse=True)
            
            print(f"✅ Stockfish found {len(moves_with_eval)} candidate moves:")
            for i, (move, eval_score) in enumerate(moves_with_eval):
                print(f"   {i+1}. {move} (eval: {eval_score/100:.2f})")
            
            return moves_with_eval
            
        except Exception as e:
            print(f"❌ Error getting moves from Stockfish: {e}")
            return []

    def parse_fen_to_board(self, fen):
        """Convert FEN string to board representation"""
        board = [['' for _ in range(8)] for _ in range(8)]
        fen_board = fen.split(' ')[0]
        rows = fen_board.split('/')
        
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    board[row_idx][col_idx] = char
                    col_idx += 1
        
        return board

    def evaluate_move_safety(self, fen, move):
        """Evaluate the safety of a move by checking for piece threats"""
        try:
            board = self.parse_fen_to_board(fen)
            from_square = move[:2]
            to_square = move[2:4]
            
            # Convert squares to board indices
            from_col = ord(from_square[0]) - ord('a')
            from_row = 8 - int(from_square[1])
            to_col = ord(to_square[0]) - ord('a')
            to_row = 8 - int(to_square[1])
            
            moving_piece = board[from_row][from_col]
            captured_piece = board[to_row][to_col]
            
            # Calculate risk and reward scores
            risk_score = 0
            reward_score = 0
            
            # Reward for capturing opponent pieces
            if captured_piece and captured_piece != '':
                piece_value = self.piece_values.get(captured_piece, 0)
                reward_score += piece_value
                print(f"   📈 Capturing {captured_piece} (+{piece_value} points)")
            
            # Risk assessment: check if moving piece will be under attack
            # Simulate the move
            temp_board = [row[:] for row in board]
            temp_board[to_row][to_col] = moving_piece
            temp_board[from_row][from_col] = ''
            
            # Check if the moved piece is now attacked by opponent
            our_color = 'white' if moving_piece.isupper() else 'black'
            opponent_attacks = self.get_attacked_squares(temp_board, 'black' if our_color == 'white' else 'white')
            
            square_key = f"{chr(to_col + ord('a'))}{8 - to_row}"
            if square_key in opponent_attacks:
                piece_value = self.piece_values.get(moving_piece, 0)
                risk_score += piece_value
                print(f"   ⚠️ {moving_piece} at {to_square} will be under attack (risk: -{piece_value} points)")
            
            # Additional strategic considerations
            
            # Penalize exposing valuable pieces unnecessarily
            if moving_piece.lower() in ['q', 'r', 'b', 'n'] and not captured_piece:
                # Check if the piece was safe before and is now in danger
                from_square_key = f"{chr(from_col + ord('a'))}{8 - from_row}"
                opponent_attacks_before = self.get_attacked_squares(board, 'black' if our_color == 'white' else 'white')
                
                if from_square_key not in opponent_attacks_before and square_key in opponent_attacks:
                    risk_score += 1  # Additional risk for unnecessary exposure
                    print(f"   🔍 Unnecessary exposure of {moving_piece}")
            
            # Bonus for developing pieces (knights and bishops from back rank)
            if moving_piece.lower() in ['n', 'b']:
                start_rank = 1 if our_color == 'white' else 8
                if int(from_square[1]) == start_rank:
                    reward_score += 0.5
                    print(f"   🎯 Developing piece (+0.5 points)")
            
            # Bonus for castling (detected by king moving 2 squares)
            if moving_piece.lower() == 'k' and abs(to_col - from_col) == 2:
                reward_score += 2
                print(f"   🏰 Castling bonus (+2 points)")
            
            # Center control bonus
            center_squares = ['d4', 'd5', 'e4', 'e5']
            if to_square in center_squares:
                reward_score += 0.3
                print(f"   🎯 Center control (+0.3 points)")
            
            safety_score = reward_score - risk_score
            print(f"   📊 Safety evaluation: {safety_score:.2f} (reward: {reward_score:.2f}, risk: {risk_score:.2f})")
            
            return safety_score
            
        except Exception as e:
            print(f"❌ Error evaluating move safety: {e}")
            return 0

    def get_attacked_squares(self, board, attacking_color):
        """Get all squares attacked by pieces of the given color"""
        attacked_squares = set()
        
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if not piece:
                    continue
                
                piece_color = 'white' if piece.isupper() else 'black'
                if piece_color != attacking_color:
                    continue
                
                # Get attacked squares for this piece
                piece_attacks = self.get_piece_attacks(board, row, col, piece.lower())
                attacked_squares.update(piece_attacks)
        
        return attacked_squares

    def get_piece_attacks(self, board, row, col, piece_type):
        """Get squares attacked by a specific piece"""
        attacks = set()
        
        if piece_type == 'p':  # Pawn
            direction = -1 if board[row][col].isupper() else 1
            # Pawn attacks diagonally
            for dc in [-1, 1]:
                new_row, new_col = row + direction, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    attacks.add(f"{chr(new_col + ord('a'))}{8 - new_row}")
        
        elif piece_type == 'r':  # Rook
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                for i in range(1, 8):
                    new_row, new_col = row + dr * i, col + dc * i
                    if not (0 <= new_row < 8 and 0 <= new_col < 8):
                        break
                    attacks.add(f"{chr(new_col + ord('a'))}{8 - new_row}")
                    if board[new_row][new_col]:  # Piece blocks further movement
                        break
        
        elif piece_type == 'n':  # Knight
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    attacks.add(f"{chr(new_col + ord('a'))}{8 - new_row}")
        
        elif piece_type == 'b':  # Bishop
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in directions:
                for i in range(1, 8):
                    new_row, new_col = row + dr * i, col + dc * i
                    if not (0 <= new_row < 8 and 0 <= new_col < 8):
                        break
                    attacks.add(f"{chr(new_col + ord('a'))}{8 - new_row}")
                    if board[new_row][new_col]:  # Piece blocks further movement
                        break
        
        elif piece_type == 'q':  # Queen (combines rook and bishop)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in directions:
                for i in range(1, 8):
                    new_row, new_col = row + dr * i, col + dc * i
                    if not (0 <= new_row < 8 and 0 <= new_col < 8):
                        break
                    attacks.add(f"{chr(new_col + ord('a'))}{8 - new_row}")
                    if board[new_row][new_col]:  # Piece blocks further movement
                        break
        
        elif piece_type == 'k':  # King
            king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dr, dc in king_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    attacks.add(f"{chr(new_col + ord('a'))}{8 - new_row}")
        
        return attacks

    def choose_best_move(self, fen, moves_with_eval):
        """Choose the best move considering both engine evaluation and safety"""
        if not moves_with_eval:
            return None
            
        print(f"\n🧠 Evaluating {len(moves_with_eval)} candidate moves:")
        
        move_scores = []
        
        for move, engine_eval in moves_with_eval:
            # Normalize engine evaluation (convert centipawns to a more manageable scale)
            normalized_engine_eval = engine_eval / 100.0
            
            # Calculate safety score
            safety_score = self.evaluate_move_safety(fen, move)
            
            # Combine scores with weights
            # Engine evaluation is primary (weight 1.0), safety is secondary (weight 0.5)
            combined_score = normalized_engine_eval + (safety_score * 0.5)
            
            move_scores.append((move, combined_score, normalized_engine_eval, safety_score))
            print(f"   🎯 {move}: Combined={combined_score:.2f} (Engine={normalized_engine_eval:.2f}, Safety={safety_score:.2f})")
        
        # Sort by combined score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_move = move_scores[0][0]
        best_score = move_scores[0][1]
        
        print(f"\n🏆 Selected move: {best_move} (score: {best_score:.2f})")
        
        return best_move

    def get_best_move(self, fen, depth=18):
        """Get the best move using enhanced evaluation"""
        # Get top 2 moves from Stockfish
        moves_with_eval = self.get_multiple_best_moves(fen, depth, 2)
        
        if not moves_with_eval:
            print("❌ No moves found from Stockfish")
            return None
        
        # Choose the best move considering safety and strategy
        best_move = self.choose_best_move(fen, moves_with_eval)
        
        return best_move
    
    def convert_move_for_black(self, move):
        """Convert move coordinates when playing as black"""
        if self.player_color == 'white':
            return move
            
        # When playing as black, we need to flip the coordinates
        # a1 becomes h8, h8 becomes a1, etc.
        def flip_square(square):
            file = square[0]
            rank = square[1]
            new_file = chr(ord('h') - (ord(file) - ord('a')))
            new_rank = str(9 - int(rank))
            return new_file + new_rank
        
        if len(move) >= 4:
            from_square = move[:2]
            to_square = move[2:4]
            flipped_from = flip_square(from_square)
            flipped_to = flip_square(to_square)
            return flipped_from + flipped_to + move[4:]  # Include promotion if present
        
        return move
    
    def make_move(self, move):
        """Execute the move on the chess board"""
        try:
            if len(move) < 4:
                print(f"❌ Invalid move format: {move}")
                return False
            
            # Convert move if playing as black
            display_move = self.convert_move_for_black(move)
            
            from_square = display_move[:2]
            to_square = display_move[2:4]
            
            if from_square not in self.square_to_coord or to_square not in self.square_to_coord:
                print(f"❌ Invalid squares: {from_square} -> {to_square}")
                return False
            
            from_x, from_y = self.square_to_coord[from_square]
            to_x, to_y = self.square_to_coord[to_square]
            
            # Perform the move
            print(f"🎯 Moving piece from {from_square} to {to_square}")
            pyautogui.moveTo(from_x, from_y)
            pyautogui.click()
            time.sleep(0.5)
            pyautogui.moveTo(to_x, to_y)
            pyautogui.click()
            time.sleep(0.2)
            pyautogui.moveTo(745, 600)
            
            print("✅ Move executed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error making move: {e}")
            return False
    
    def run(self):
        """Main game loop"""
        print("🚀 Enhanced Chess Bot Started!")
        print("🔍 Detecting player color...")
        
        # Detect player color at startup
        self.player_color = self.detect_player_color()
        if not self.player_color:
            print("❌ Could not detect player color. Please restart the bot.")
            return
        
        print("🧠 Bot will now analyze multiple moves and choose the safest option!")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Check whose turn it is
                turn = self.detect_turn()
                
                if turn == "your":
                    print(f"\n🟢 It's your turn! (Playing as {self.player_color.upper()}) Processing move...")
                    
                    # Step 1: Capture board
                    if not self.capture_board():
                        time.sleep(2)
                        continue
                    
                    # Step 2: Generate FEN
                    fen = self.generate_fen()
                    if not fen:
                        time.sleep(2)
                        continue
                    
                    # Step 3: Get best move with enhanced evaluation
                    best_move = self.get_best_move(fen)
                    if not best_move:
                        time.sleep(2)
                        continue
                    
                    # Step 4: Execute the move
                    if self.make_move(best_move):
                        print("✅ Move completed! Waiting for opponent...")
                        time.sleep(3)  # Wait a bit before checking turn again
                    else:
                        time.sleep(2)
                
                elif turn == "opponent":
                    print("🔴 Opponent's turn, waiting...")
                    time.sleep(1)
                
                else:
                    print("⚠️ Turn detection unclear, retrying...")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Enhanced Chess Bot stopped by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    bot = ChessBot()
    bot.run()