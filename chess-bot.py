import pyautogui
import time
import cv2
import numpy as np
import os
import subprocess
import random
import json
import chess
import chess.pgn
import io
from datetime import datetime

class EnhancedChessBot:
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
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.board_path = os.path.join(self.base_path, 'board.png')
        self.stockfish_path = os.path.join(self.base_path, 'stockfish.exe')
        self.opening_book_path = os.path.join(self.base_path, 'opening_book.json')
        
        # Player color (will be detected automatically)
        self.player_color = None
        
        # Game state tracking
        self.move_history = []
        self.position_history = {}  # For threefold repetition detection
        self.game_phase = "opening"  # opening, middlegame, endgame
        self.total_pieces = 32
        self.remaining_time = 180  # Default 3 minutes
        self.opponent_time = 180
        self.last_move_time = time.time()
        
        # Piece values for evaluation (standard + positional bonus)
        self.piece_values = {
            'p': 1, 'P': 1,    # Pawn
            'n': 3, 'N': 3,    # Knight
            'b': 3.25, 'B': 3.25,  # Bishop (slightly higher than knight)
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
        
        # Positional piece-square tables (higher values are better positions)
        self.piece_square_tables = self.initialize_piece_square_tables()
        
        # Common tactical patterns
        self.tactical_patterns = self.initialize_tactical_patterns()
        
        # Opening book
        self.opening_book = self.load_opening_book()
        
        # Load templates
        self.load_templates()
        
        # Initialize chess board for internal representation
        self.board = chess.Board()
        
        # Performance metrics
        self.games_played = 0
        self.games_won = 0
        self.avg_move_time = 0
        
        # Adaptive parameters
        self.depth_by_phase = {
            "opening": 15,     # Standard depth for opening
            "middlegame": 18,  # Deeper search for middlegame
            "endgame":18,     # Even deeper for endgame
            "critical": 20     # For critical positions
        }
        
        # Time management
        self.time_allocation = {
            "opening": 0.05,   # 5% of time for opening moves
            "middlegame": 0.6, # 60% of time for middlegame
            "endgame": 0.35    # 35% of time for endgame
        }
        
        print(f"🚀 Enhanced Chess Bot v3 initialized!")
    
    def initialize_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        tables = {}
        
        # Pawns: encourage center control, advancement
        tables['P'] = [
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [ 5,  5, 10, 25, 25, 10,  5,  5],
            [ 0,  0,  0, 20, 20,  0,  0,  0],
            [ 5, -5,-10,  0,  0,-10, -5,  5],
            [ 5, 10, 10,-20,-20, 10, 10,  5],
            [ 0,  0,  0,  0,  0,  0,  0,  0]
        ]
        
        # Knights: prefer center positions, avoid edges
        tables['N'] = [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ]
        
        # Bishops: prefer diagonals, avoid corners
        tables['B'] = [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5,  5,  5,  5,  5,-10],
            [-10,  0,  5,  0,  0,  5,  0,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ]
        
        # Rooks: prefer open files, 7th rank
        tables['R'] = [
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [ 0,  0,  0,  5,  5,  0,  0,  0]
        ]
        
        # Queens: combination of rook and bishop mobility
        tables['Q'] = [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [ -5,  0,  5,  5,  5,  5,  0, -5],
            [  0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]
        
        # King (middlegame): seek shelter, avoid center
        tables['K_mg'] = [
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [ 20, 20,  0,  0,  0,  0, 20, 20],
            [ 20, 30, 10,  0,  0, 10, 30, 20]
        ]
        
        # King (endgame): seek center
        tables['K_eg'] = [
            [-50,-40,-30,-20,-20,-30,-40,-50],
            [-30,-20,-10,  0,  0,-10,-20,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-30,  0,  0,  0,  0,-30,-30],
            [-50,-30,-30,-30,-30,-30,-30,-50]
        ]
        
        # Create tables for black pieces (mirror of white tables)
        for piece, table in list(tables.items()):
            if piece != 'K_mg' and piece != 'K_eg':
                tables[piece.lower()] = list(reversed(table))
            
        # Special case for kings
        tables['k_mg'] = list(reversed(tables['K_mg']))
        tables['k_eg'] = list(reversed(tables['K_eg']))
        
        return tables
    
    def initialize_tactical_patterns(self):
        """Initialize common tactical patterns to recognize"""
        patterns = {
            "fork": {
                "description": "One piece attacking multiple enemy pieces",
                "detection": self.detect_fork
            },
            "pin": {
                "description": "A piece cannot move because it would expose a more valuable piece",
                "detection": self.detect_pin
            },
            "discovered_attack": {
                "description": "Moving one piece reveals an attack by another",
                "detection": self.detect_discovered_attack
            },
            "skewer": {
                "description": "Attack on two pieces in a line, first piece must move revealing second",
                "detection": self.detect_skewer
            },
            "hanging_piece": {
                "description": "Undefended piece that can be captured",
                "detection": self.detect_hanging_piece
            },
            "trapped_piece": {
                "description": "Piece with limited or no mobility",
                "detection": self.detect_trapped_piece
            }
        }
        return patterns
    
    def detect_fork(self, board, move):
        """Detect if a move creates a fork (attacking multiple pieces)"""
        try:
            # Convert move to chess.Move
            uci_move = chess.Move.from_uci(move)
            
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(uci_move)
            
            # Get the piece that moved
            piece = board_copy.piece_at(uci_move.to_square)
            if not piece:
                return False, 0
                
            # Get all squares attacked by this piece
            attacked_squares = set()
            for square in chess.SQUARES:
                if board_copy.is_attacked_by(not piece.color, square) and board_copy.piece_at(square) and board_copy.piece_at(square).color == piece.color:
                    attacked_squares.add(square)
            
            # Count valuable pieces under attack (not pawns)
            valuable_targets = 0
            for square in attacked_squares:
                target = board_copy.piece_at(square)
                if target and target.color != piece.color and target.piece_type != chess.PAWN:
                    valuable_targets += 1
            
            # It's a fork if attacking 2+ valuable pieces
            is_fork = valuable_targets >= 2
            return is_fork, valuable_targets
            
        except Exception as e:
            print(f"❌ Error detecting fork: {e}")
            return False, 0
    
    def detect_pin(self, board, move):
        """Detect if a move creates a pin"""
        try:
            # Convert move to chess.Move
            uci_move = chess.Move.from_uci(move)
            
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(uci_move)
            
            # Check for pins
            pins = 0
            for square in chess.SQUARES:
                piece = board_copy.piece_at(square)
                if not piece or piece.color != board_copy.turn:
                    continue
                
                # Check if this piece is pinned
                if board_copy.is_pinned(board_copy.turn, square):
                    pins += 1
            
            return pins > 0, pins
            
        except Exception as e:
            print(f"❌ Error detecting pin: {e}")
            return False, 0
    
    def detect_discovered_attack(self, board, move):
        """Detect if a move creates a discovered attack"""
        try:
            # Convert move to chess.Move
            uci_move = chess.Move.from_uci(move)
            
            # Get the piece that will move
            piece = board.piece_at(uci_move.from_square)
            if not piece:
                return False, 0
                
            # Get all squares attacked before the move
            attacked_before = set()
            for square in chess.SQUARES:
                if board.is_attacked_by(piece.color, square):
                    attacked_before.add(square)
            
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(uci_move)
            
            # Get all squares attacked after the move
            attacked_after = set()
            for square in chess.SQUARES:
                if board_copy.is_attacked_by(not board_copy.turn, square):
                    attacked_after.add(square)
            
            # New attacks that weren't possible before
            new_attacks = attacked_after - attacked_before
            
            # Count valuable pieces under new attack
            valuable_targets = 0
            for square in new_attacks:
                target = board_copy.piece_at(square)
                if target and target.color != piece.color and target.piece_type != chess.PAWN:
                    valuable_targets += 1
            
            return len(new_attacks) > 0, valuable_targets
            
        except Exception as e:
            print(f"❌ Error detecting discovered attack: {e}")
            return False, 0
    
    def detect_skewer(self, board, move):
        """Detect if a move creates a skewer"""
        try:
            # Convert move to chess.Move
            uci_move = chess.Move.from_uci(move)
            
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(uci_move)
            
            # Get the piece that moved
            piece = board_copy.piece_at(uci_move.to_square)
            if not piece:
                return False, 0
                
            # Only sliding pieces can create skewers
            if piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                return False, 0
            
            # Check for potential skewers along the appropriate lines
            directions = []
            if piece.piece_type in [chess.ROOK, chess.QUEEN]:
                directions.extend([(0, 1), (1, 0), (0, -1), (-1, 0)])  # Horizontal/vertical
            if piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])  # Diagonal
            
            # Convert to_square to coordinates
            to_file = chess.square_file(uci_move.to_square)
            to_rank = chess.square_rank(uci_move.to_square)
            
            for d_file, d_rank in directions:
                # Look for two pieces along this direction
                pieces_found = []
                for i in range(1, 8):
                    new_file = to_file + d_file * i
                    new_rank = to_rank + d_rank * i
                    
                    if not (0 <= new_file < 8 and 0 <= new_rank < 8):
                        break
                        
                    new_square = chess.square(new_file, new_rank)
                    target = board_copy.piece_at(new_square)
                    
                    if target:
                        if target.color != piece.color:
                            pieces_found.append((target, new_square))
                        else:
                            break  # Blocked by own piece
                        
                        if len(pieces_found) == 2:
                            break  # Found two pieces
                
                # Check if we found two enemy pieces and the first is more valuable
                if len(pieces_found) == 2:
                    first_piece, _ = pieces_found[0]
                    second_piece, _ = pieces_found[1]
                    
                    if first_piece.piece_type > second_piece.piece_type:
                        return True, 1
            
            return False, 0
            
        except Exception as e:
            print(f"❌ Error detecting skewer: {e}")
            return False, 0
    
    def detect_hanging_piece(self, board, move):
        """Detect if a move allows capturing a hanging piece"""
        try:
            # Convert move to chess.Move
            uci_move = chess.Move.from_uci(move)
            
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(uci_move)
            
            # Check for hanging pieces (attacked but not defended)
            hanging_value = 0
            
            for square in chess.SQUARES:
                piece = board_copy.piece_at(square)
                if not piece or piece.color == board_copy.turn:
                    continue
                
                # Check if piece is attacked
                if board_copy.is_attacked_by(board_copy.turn, square):
                    # Check if piece is defended
                    if not board_copy.is_attacked_by(not board_copy.turn, square):
                        # Hanging piece found!
                        piece_value = {
                            chess.PAWN: 1,
                            chess.KNIGHT: 3,
                            chess.BISHOP: 3,
                            chess.ROOK: 5,
                            chess.QUEEN: 9,
                            chess.KING: 100
                        }
                        hanging_value += piece_value.get(piece.piece_type, 0)
            
            return hanging_value > 0, hanging_value
            
        except Exception as e:
            print(f"❌ Error detecting hanging piece: {e}")
            return False, 0
    
    def detect_trapped_piece(self, board, move):
        """Detect if a move traps an opponent's piece"""
        try:
            # Convert move to chess.Move
            uci_move = chess.Move.from_uci(move)
            
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(uci_move)
            
            # Check for trapped pieces (no legal moves)
            trapped_value = 0
            
            for square in chess.SQUARES:
                piece = board_copy.piece_at(square)
                if not piece or piece.color != board_copy.turn:
                    continue
                
                # Count legal moves for this piece
                legal_moves = 0
                for move in board_copy.legal_moves:
                    if move.from_square == square:
                        legal_moves += 1
                
                # If no legal moves, piece is trapped
                if legal_moves == 0:
                    piece_value = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                        chess.KING: 0  # King can't be trapped without being in check
                    }
                    trapped_value += piece_value.get(piece.piece_type, 0)
            
            return trapped_value > 0, trapped_value
            
        except Exception as e:
            print(f"❌ Error detecting trapped piece: {e}")
            return False, 0
    
    def load_opening_book(self):
        """Load opening book from file or create default if not exists"""
        try:
            if os.path.exists(self.opening_book_path):
                with open(self.opening_book_path, 'r') as f:
                    return json.load(f)
            else:
                # Create a basic opening book with common openings
                opening_book = {
                    # Starting position
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": {
                        "e2e4": 40,  # King's Pawn Opening
                        "d2d4": 35,  # Queen's Pawn Opening
                        "c2c4": 15,  # English Opening
                        "g1f3": 10   # Réti Opening
                    },
                    # After 1.e4
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3": {
                        "e7e5": 40,  # Open Game
                        "c7c5": 30,  # Sicilian Defense
                        "e7e6": 15,  # French Defense
                        "c7c6": 10,  # Caro-Kann Defense
                        "d7d5": 5    # Scandinavian Defense
                    },
                    # After 1.d4
                    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3": {
                        "d7d5": 40,  # Closed Game
                        "g8f6": 30,  # Indian Defense
                        "e7e6": 15,  # French-like setup
                        "c7c5": 10,  # Benoni Defense
                        "f7f5": 5    # Dutch Defense
                    },
                    # Sicilian Defense
                    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6": {
                        "g1f3": 40,  # Open Sicilian
                        "b1c3": 30,  # Closed Sicilian
                        "c2c3": 20,  # Alapin
                        "c2c4": 10   # Grand Prix Attack
                    },
                    # French Defense
                    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq -": {
                        "e4e5": 40,  # Advance Variation
                        "e4d5": 30,  # Exchange Variation
                        "b1c3": 20,  # Classical Variation
                        "b1d2": 10   # Tarrasch Variation
                    },
                    # Ruy Lopez
                    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": {
                        "a7a6": 50,  # Morphy Defense
                        "g8f6": 20,  # Berlin Defense
                        "f8c5": 15,  # Classical Defense
                        "g7g6": 10,  # Fianchetto Defense
                        "f7f5": 5    # Schliemann Defense
                    }
                    # More positions can be added
                }
                
                # Save the opening book
                with open(self.opening_book_path, 'w') as f:
                    json.dump(opening_book, f, indent=2)
                
                return opening_book
        except Exception as e:
            print(f"❌ Error loading opening book: {e}")
            return {}
    
    def update_opening_book(self, fen_key, move, result=None):
        """Update opening book with new move and result"""
        try:
            # Simplify FEN to use as key (remove move counters)
            fen_parts = fen_key.split(' ')
            if len(fen_parts) > 4:
                fen_key = ' '.join(fen_parts[:4])
            
            # Add position if not exists
            if fen_key not in self.opening_book:
                self.opening_book[fen_key] = {}
            
            # Add or update move
            if move in self.opening_book[fen_key]:
                # Increase weight based on result
                if result == "win":
                    self.opening_book[fen_key][move] += 5
                elif result == "draw":
                    self.opening_book[fen_key][move] += 1
            else:
                # New move starts with weight 5
                self.opening_book[fen_key][move] = 5
            
            # Save updated opening book
            with open(self.opening_book_path, 'w') as f:
                json.dump(self.opening_book, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error updating opening book: {e}")
    
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
        template_dir = os.path.join(self.base_path, 'templates')
        
        template_names = [
            'wp_w', 'wp_b', 'bp_w', 'bp_b', 'wr_w', 'wr_b', 'br_w', 'br_b',
            'wn_w', 'wn_b', 'bn_w', 'bn_b', 'wb_w', 'wb_b', 'bb_w', 'bb_b',
            'wq_w', 'wq_b', 'bq_w', 'bq_b', 'wk_w', 'wk_b', 'bk_w', 'bk_b'
        ]
        
        for name in template_names:
            template_path = os.path.join(template_dir, f'{name}.png')
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
            
            # Update internal board representation
            try:
                self.board = chess.Board(fen)
            except ValueError:
                print("⚠️ Invalid FEN, using internal board state")
            
            # Update game phase based on piece count
            self.update_game_phase()
            
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
        active_color = 'w' if self.detect_turn() == "your" else 'b'
        
        # Simplified castling rights (assume all available)
        castling = 'KQkq'
        
        # No en passant target square
        en_passant = '-'
        
        return '/'.join(fen_rows) + f' {active_color} {castling} {en_passant} 0 1'
    
    def update_game_phase(self):
        """Update the game phase based on piece count and position"""
        try:
            # Count pieces on the board
            piece_count = 0
            for square in chess.SQUARES:
                if self.board.piece_at(square) is not None:
                    piece_count += 1
            
            # Update total pieces
            self.total_pieces = piece_count
            
            # Determine game phase
            if piece_count >= 24:  # 75% of pieces remain
                self.game_phase = "opening"
            elif piece_count >= 12:  # 37.5% of pieces remain
                self.game_phase = "middlegame"
            else:
                self.game_phase = "endgame"
                
            print(f"🎮 Game phase: {self.game_phase.upper()} (pieces: {piece_count})")
            
        except Exception as e:
            print(f"❌ Error updating game phase: {e}")
    
    def is_critical_position(self):
        """Determine if the current position is critical and needs deeper analysis"""
        try:
            # Check if king is in check
            if self.board.is_check():
                return True
                
            # Check if there's a potential checkmate threat
            # (opponent has a piece attacking near our king)
            king_square = self.board.king(self.board.turn)
            if king_square:
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                
                # Check squares around king
                for df in [-1, 0, 1]:
                    for dr in [-1, 0, 1]:
                        if df == 0 and dr == 0:
                            continue  # Skip king's square
                            
                        new_file = king_file + df
                        new_rank = king_rank + dr
                        
                        if 0 <= new_file < 8 and 0 <= new_rank < 8:
                            square = chess.square(new_file, new_rank)
                            if self.board.is_attacked_by(not self.board.turn, square):
                                return True
            
            # Check for hanging pieces
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    if self.board.is_attacked_by(not self.board.turn, square) and not self.board.is_attacked_by(self.board.turn, square):
                        # Valuable piece is hanging
                        if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            return True
            
            # Check for potential captures of valuable pieces
            for move in self.board.legal_moves:
                if self.board.is_capture(move):
                    captured_piece = self.board.piece_at(move.to_square)
                    if captured_piece and captured_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error checking critical position: {e}")
            return False
    
    def get_multiple_best_moves(self, fen, depth=18, num_moves=3):
        """Get multiple best moves from Stockfish with their evaluations"""
        try:
            # Adjust depth based on game phase and position criticality
            if self.is_critical_position():
                depth = self.depth_by_phase["critical"]
                print(f"⚠️ Critical position detected! Increasing depth to {depth}")
            else:
                depth = self.depth_by_phase[self.game_phase]
            
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Set options for stronger play
            commands = [
                'setoption name Threads value 4',  # Use 4 CPU threads
                'setoption name Hash value 128',   # Use 128MB hash table
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
    
    def check_opening_book(self, fen):
        """Check if current position is in opening book and return move if found"""
        try:
            # Simplify FEN to use as key (remove move counters)
            fen_parts = fen.split(' ')
            if len(fen_parts) > 4:
                fen_key = ' '.join(fen_parts[:4])
            else:
                fen_key = fen
            
            # Check if position is in opening book
            if fen_key in self.opening_book and self.game_phase == "opening":
                moves = self.opening_book[fen_key]
                if moves:
                    # Select move based on weights
                    total_weight = sum(moves.values())
                    r = random.randint(1, total_weight)
                    
                    cumulative = 0
                    for move, weight in moves.items():
                        cumulative += weight
                        if r <= cumulative:
                            print(f"📚 Found move in opening book: {move}")
                            return move
            
            return None
            
        except Exception as e:
            print(f"❌ Error checking opening book: {e}")
            return None
    
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
    
    def evaluate_positional_score(self, fen, move):
        """Evaluate the positional score of a move using piece-square tables"""
        try:
            # Parse FEN to board
            board = self.parse_fen_to_board(fen)
            
            # Parse move
            from_square = move[:2]
            to_square = move[2:4]
            
            # Convert algebraic notation to indices
            from_col = ord(from_square[0]) - ord('a')
            from_row = 8 - int(from_square[1])
            to_col = ord(to_square[0]) - ord('a')
            to_row = 8 - int(to_square[1])
            
            # Get the moving piece
            piece = board[from_row][from_col]
            if not piece:
                return 0
            
            # Calculate positional change
            old_score = 0
            new_score = 0
            
            # Get the appropriate piece-square table
            if piece.lower() == 'k':
                # Use different tables for king in different game phases
                if self.game_phase == "endgame":
                    table_key = f"{piece}_eg"
                else:
                    table_key = f"{piece}_mg"
            else:
                table_key = piece
            
            if table_key in self.piece_square_tables:
                old_score = self.piece_square_tables[table_key][from_row][from_col]
                new_score = self.piece_square_tables[table_key][to_row][to_col]
            
            # Calculate positional improvement
            positional_change = new_score - old_score
            
            # Additional bonuses
            bonus = 0
            
            # Bonus for controlling center with pawns
            if piece.lower() == 'p' and to_square in ['d4', 'd5', 'e4', 'e5']:
                bonus += 10
            
            # Bonus for developing pieces in opening
            if self.game_phase == "opening" and piece.lower() in ['n', 'b']:
                if (piece.isupper() and from_row == 7) or (not piece.isupper() and from_row == 0):
                    bonus += 15
            
            # Bonus for rook on open file
            if piece.lower() == 'r':
                open_file = True
                for row in range(8):
                    if row != from_row and board[row][to_col].lower() == 'p':
                        open_file = False
                        break
                if open_file:
                    bonus += 20
            
            # Bonus for bishop pair
            if piece.lower() == 'b':
                bishop_count = 0
                for r in range(8):
                    for c in range(8):
                        if board[r][c] == piece:
                            bishop_count += 1
                if bishop_count >= 2:
                    bonus += 15
            
            # Penalty for moving the same piece multiple times in opening
            if self.game_phase == "opening" and len(self.move_history) < 10:
                piece_moves = 0
                for prev_move in self.move_history[-4:]:  # Check last 4 moves
                    if prev_move.startswith(from_square):
                        piece_moves += 1
                if piece_moves > 0:
                    bonus -= 10 * piece_moves
            
            total_score = positional_change + bonus
            print(f"   📊 Positional evaluation: {total_score} (change: {positional_change}, bonus: {bonus})")
            
            return total_score
            
        except Exception as e:
            print(f"❌ Error evaluating positional score: {e}")
            return 0
    
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
                
                # Check if the piece is defended
                our_defenses = self.get_attacked_squares(temp_board, our_color)
                if square_key in our_defenses:
                    # Piece is defended, reduce risk
                    risk_score += piece_value * 0.5
                    print(f"   ⚠️ {moving_piece} at {to_square} will be under attack but defended (risk: -{piece_value * 0.5} points)")
                else:
                    # Piece is not defended
                    risk_score += piece_value
                    print(f"   ⚠️ {moving_piece} at {to_square} will be under attack and undefended (risk: -{piece_value} points)")
            
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
            
            # Pawn structure bonuses/penalties
            if moving_piece.lower() == 'p':
                # Check for isolated pawns
                files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                to_file_idx = files.index(to_square[0])
                
                # Check adjacent files for friendly pawns
                has_adjacent_pawn = False
                for adj_file_idx in [to_file_idx - 1, to_file_idx + 1]:
                    if 0 <= adj_file_idx < 8:
                        adj_file = files[adj_file_idx]
                        for rank in range(1, 9):
                            check_square = f"{adj_file}{rank}"
                            # Convert to board indices
                            check_col = ord(adj_file) - ord('a')
                            check_row = 8 - rank
                            if 0 <= check_row < 8 and 0 <= check_col < 8:
                                if temp_board[check_row][check_col].lower() == 'p' and temp_board[check_row][check_col].isupper() == moving_piece.isupper():
                                    has_adjacent_pawn = True
                                    break
                
                if not has_adjacent_pawn:
                    risk_score += 0.2
                    print(f"   ⚠️ Creating isolated pawn (-0.2 points)")
                
                # Bonus for passed pawns
                is_passed = True
                forward_direction = -1 if moving_piece.isupper() else 1
                for check_row in range(to_row + forward_direction, 0 if forward_direction < 0 else 8, forward_direction):
                    for check_col in [to_col - 1, to_col, to_col + 1]:
                        if 0 <= check_col < 8:
                            if temp_board[check_row][check_col].lower() == 'p' and temp_board[check_row][check_col].isupper() != moving_piece.isupper():
                                is_passed = False
                                break
                
                if is_passed:
                    # Bonus increases as pawn advances
                    rank_bonus = abs(to_row - (7 if moving_piece.isupper() else 0)) * 0.1
                    reward_score += 0.5 + rank_bonus
                    print(f"   🎯 Passed pawn (+{0.5 + rank_bonus:.1f} points)")
            
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
    
    def evaluate_tactical_opportunities(self, fen, move):
        """Evaluate tactical opportunities in a position"""
        try:
            # Convert FEN to chess.Board
            board = chess.Board(fen)
            
            # Initialize tactical score
            tactical_score = 0
            
            # Check for tactical patterns
            for pattern_name, pattern_info in self.tactical_patterns.items():
                is_present, value = pattern_info["detection"](board, move)
                if is_present:
                    print(f"   🎯 Tactical opportunity: {pattern_name} (value: {value})")
                    tactical_score += value
            
            return tactical_score
            
        except Exception as e:
            print(f"❌ Error evaluating tactical opportunities: {e}")
            return 0
    
    def choose_best_move(self, fen, moves_with_eval):
        """Choose the best move considering multiple factors"""
        if not moves_with_eval:
            return None
            
        print(f"\n🧠 Evaluating {len(moves_with_eval)} candidate moves:")
        
        move_scores = []
        
        for move, engine_eval in moves_with_eval:
            # Normalize engine evaluation (convert centipawns to a more manageable scale)
            normalized_engine_eval = engine_eval / 100.0
            
            # Calculate safety score
            safety_score = self.evaluate_move_safety(fen, move)
            
            # Calculate positional score
            positional_score = self.evaluate_positional_score(fen, move)
            
            # Calculate tactical score
            tactical_score = self.evaluate_tactical_opportunities(fen, move)
            
            # Combine scores with weights that vary by game phase
            if self.game_phase == "opening":
                # In opening, prioritize development and position
                engine_weight = 0.6
                safety_weight = 0.5
                positional_weight = 0.8
                tactical_weight = 0.7
            elif self.game_phase == "middlegame":
                # In middlegame, prioritize tactics and engine eval
                engine_weight = 0.8
                safety_weight = 0.6
                positional_weight = 0.5
                tactical_weight = 0.9
            else:  # endgame
                # In endgame, prioritize engine eval and safety
                engine_weight = 1.0
                safety_weight = 0.7
                positional_weight = 0.4
                tactical_weight = 0.6
            
            # Calculate combined score
            combined_score = (
                normalized_engine_eval * engine_weight +
                safety_score * safety_weight +
                positional_score * positional_weight / 100 +  # Scale down positional score
                tactical_score * tactical_weight
            )
            
            move_scores.append((
                move, 
                combined_score, 
                normalized_engine_eval, 
                safety_score, 
                positional_score, 
                tactical_score
            ))
            
            print(f"   🎯 {move}: Combined={combined_score:.2f} (Engine={normalized_engine_eval:.2f}*{engine_weight}, Safety={safety_score:.2f}*{safety_weight}, Position={positional_score:.2f}*{positional_weight/100}, Tactical={tactical_score:.2f}*{tactical_weight})")
        
        # Sort by combined score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_move = move_scores[0][0]
        best_score = move_scores[0][1]
        
        print(f"\n🏆 Selected move: {best_move} (score: {best_score:.2f})")
        
        return best_move
    
    def get_best_move(self, fen):
        """Get the best move using enhanced evaluation"""
        # First check opening book
        book_move = self.check_opening_book(fen)
        if book_move:
            # Add to move history
            self.move_history.append(book_move)
            return book_move
        
        # Get multiple candidate moves from Stockfish
        # Adjust number of candidate moves based on game phase
        num_candidates = 3
        if self.game_phase == "critical" or self.is_critical_position():
            num_candidates = 2  # Fewer candidates in critical positions
        
        # Get depth based on game phase
        depth = self.depth_by_phase[self.game_phase]
        
        # Get candidate moves
        moves_with_eval = self.get_multiple_best_moves(fen, depth, num_candidates)
        
        if not moves_with_eval:
            print("❌ No moves found from Stockfish")
            return None
        
        # Choose the best move considering safety, position, and tactics
        best_move = self.choose_best_move(fen, moves_with_eval)
        
        # Add to move history
        if best_move:
            self.move_history.append(best_move)
            
            # Update position history for threefold repetition detection
            fen_key = ' '.join(fen.split(' ')[:4])  # Use position only, not full FEN
            if fen_key in self.position_history:
                self.position_history[fen_key] += 1
            else:
                self.position_history[fen_key] = 1
        
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
        print("🚀 Enhanced Chess Bot v3 Started!")
        print("🔍 Detecting player color...")
        
        # Detect player color at startup
        self.player_color = self.detect_player_color()
        if not self.player_color:
            print("❌ Could not detect player color. Please restart the bot.")
            return
        
        print(f"🧠 Bot will now play as {self.player_color.upper()} with advanced strategies!")
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
    bot = EnhancedChessBot()
    bot.run()
