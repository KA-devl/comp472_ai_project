from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from time import sleep
from typing import Tuple, Iterable, ClassVar, Optional, TextIO

import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class MoveType(Enum):
    MOVE = 0
    ATTACK = 1
    REPAIR = 2
    SELF_DESTRUCT = 3
    INVALID = 4


class UnitType(Enum):
    """Every unit type."""
    AI = 0  # TODO: limited
    Tech = 1
    Virus = 2
    Program = 3  # TODO: limited
    Firewall = 4  # TODO: limited


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

    @staticmethod
    def get_game_type(game_type: str) -> GameType:
        if game_type == "attacker":
            return GameType.AttackerVsComp
        elif game_type == "defender":
            return GameType.CompVsDefender
        elif game_type == "manual":
            return GameType.AttackerVsDefender
        else:
            return GameType.CompVsComp


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_same_team(self, unit: Unit) -> bool:
        return self.player == unit.player

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_amount: int):
        """Modify this unit's health by delta amount."""
        self.health += health_amount
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)  # source of the move
    dst: Coord = field(default_factory=Coord)  # destination of the move

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    ###################### HELPERS ##################################
    def is_left_top(self) -> bool:  # helper function for is_valid_move()
        """Checks whether dst coord is at the top and left of src coord"""
        return (self.dst.row < self.src.row and self.dst.col == self.src.col) or \
            (self.dst.row == self.src.row and self.dst.col < self.src.col)

    def is_adjacent(self) -> bool:
        """Check if the 2 Coords are 4-way adjacent."""
        for adj in self.src.iter_adjacent():
            if self.dst == adj:
                return True
        return False

    def is_diagonal(self) -> bool:
        """Checks if destination and source are diagonal to each other"""
        if self.src.row != self.dst.row and self.src.col != self.dst.col:
            return True
        return False

    ##################################################################################

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsComp
    alpha_beta: bool = False  # TO-DO
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    output_file: Optional[TextIO] = None

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    ############################################## MODIFIED HELPER RULE METHODS FROM D1 ##############################################
    def mod_health(self, coord: Coord, health_amount: int):
        """Modify health of unit at Coord (positive or negative amount)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_amount)
            self.remove_dead(coord)

    def is_in_combat(self, src_coord: Coord):
        """Verify if current unit is in combat with an enemy unit"""
        src_unit = self.get(src_coord)
        for adj in src_coord.iter_adjacent():
            adj_unit = self.get(adj)
            if adj_unit is not None:
                if src_unit.player != adj_unit.player:
                    return True
        return False

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False
        # if both coords are equal then it's a valid self-destruction move
        if coords.src == coords.dst:
            return True
        if coords.is_diagonal():
            return False
        if not coords.is_adjacent():
            return False
        # Virus and Tech can move freely
        if unit.type.name in {"Virus", "Tech"}:
            return True
        # Rules for program/ai/firewall
        # Checks if in combat
        is_in_combat = self.is_in_combat(coords.src)
        # Checks if destination is empty
        is_empty_destination = self.get(coords.dst) is None
        # Check if can only perform up or left
        up_or_left = coords.is_left_top()
        if unit.player == Player.Attacker:
            if is_empty_destination and (is_in_combat or not up_or_left):
                return False
        elif unit.player == Player.Defender:
            if is_empty_destination and (is_in_combat or up_or_left):
                return False
        return True

    def self_destruct(self, src_coord: Coord):  # helper function for perform_move()
        """Self destruct & damage adjacent units by 2 or 9 if they are AI"""
        for coord in src_coord.iter_range(dist=1):
            if self.get(coord):
                self.mod_health(coord, -2)
        self.mod_health(src_coord, -9)

    def perform_move(self, coords: CoordPair, is_game_clone) -> Tuple[bool, str]:
        """Performing a move depending on the move type & validity of the move"""
        if self.is_valid_move(coords):
            text_trace = ""
            source = self.get(coords.src)
            destination = self.get(coords.dst)
            # SELF-DESTRUCT
            if source == destination:
                self.self_destruct(coords.src)
                # If the game is not cloned, then we can print the game
                if not is_game_clone:
                    text_trace += f"SELF DESTRUCTION at {coords.src}\n{self}\n\n"
                    print(text_trace)
                    self.write_output(text_trace)
                return True, ""
            # MOVE
            elif destination is None:
                self.set(coords.dst, source)
                self.set(coords.src, None)
                # If the game is not cloned, then we can print the game
                if not is_game_clone:
                    text_trace += f"MOVE from : {coords.src} to {coords.dst}\n{self}\n\n"
                    print(text_trace)
                    self.write_output(text_trace)
                return True, ""
            # ATTACK
            elif source.player != destination.player:
                health_amount_destination = -source.damage_amount(destination)
                health_amount_source = -destination.damage_amount(source)
                self.mod_health(coords.src, health_amount=health_amount_source)
                self.mod_health(coords.dst, health_amount=health_amount_destination)
                if not is_game_clone:
                    text_trace += f"Attack from : {coords.src} to {coords.dst}\n{self}\n\n"
                    print(text_trace)
                    self.write_output(text_trace)
                return True, ""
            # case: action is repair
            else:
                health_amount = source.repair_amount(destination)
                if health_amount == 0:
                    # repair invalid
                    return False, "invalid move"
                self.mod_health(coords.dst, health_amount=health_amount)
                text_trace += f"Repair from {coords.src} to {coords.dst}\n{self}\n\n"
                return True, ""
        return False, "invalid move"

    def get_unit_count(self, player: Player, unit_type: UnitType) -> int:  # for e0
        """Returns the count of a specific unit type for a given player."""
        count = 0
        for _, unit in self.player_units(player):
            if unit.type == unit_type:
                count += 1
        return count

    def get_aggregate_health(self, player: Player):  # for e1
        """Get the health difference of the player's units"""
        total_health = 0
        for _, unit in self.player_units(player):
            if unit.type == UnitType.AI:
                total_health += 10 * unit.health
            else:
                total_health += unit.health
        return total_health

    def get_potential_damage_delta(self):  # for e1
        """Calculate the potential damage one type of adversary can inflict on the other"""
        attacker_potential_damage = 0
        for _, unit in self.player_units(Player.Attacker):
            for _, opp_unit in self.player_units(Player.Defender):
                attacker_potential_damage += Unit.damage_table[unit.type.value][opp_unit.type.value]

        defender_potential_damage = 0
        for _, unit in self.player_units(Player.Defender):
            for _, opp_unit in self.player_units(Player.Attacker):
                defender_potential_damage += Unit.damage_table[unit.type.value][opp_unit.type.value]

        return attacker_potential_damage - defender_potential_damage

    ######################################################################################################################################
    def start_logging(self, file_path: str) -> None:
        """Open the output file for logging."""
        self.output_file = open(file_path, 'w')

    def stop_logging(self) -> None:
        """Close the output file."""
        if self.output_file:
            self.output_file.close()
            self.output_file = None

    def write_output(self, output: str) -> None:
        """Write the output to the specified output file or sys.stdout."""
        if self.output_file:
            self.output_file.write(output)
        else:
            print(output, end='')

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        # Omit output if turns played is 0
        if self.turns_played != 0:
            output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        is_game_clone = False
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv, is_game_clone)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv, is_game_clone)
                if success:
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        is_game_clone = False
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv, is_game_clone)
            if success:
                print(f"---- Computer {self.next_player.name} ---- ", end='')
                print(result)
                self.write_output(f"Computer {self.next_player.name}: {result}\n")
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield coord, unit

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return 0, move_candidates[0], 1
        else:
            return 0, None, 0

    def evaluate(self, heuristic_type: str) -> int:
        """Evaluate the board state using the specified heuristic."""
        heuristic_value = 0
        if heuristic_type == 'e0':
            # For Attacker
            vp_attacker = self.get_unit_count(Player.Attacker, UnitType.Virus)
            tp_attacker = self.get_unit_count(Player.Attacker, UnitType.Tech)
            fp_attacker = self.get_unit_count(Player.Attacker, UnitType.Firewall)
            pp_attacker = self.get_unit_count(Player.Attacker, UnitType.Program)
            ai_attacker = self.get_unit_count(Player.Attacker, UnitType.AI)
            # For Defender
            vp_defender = self.get_unit_count(Player.Defender, UnitType.Virus)
            tp_defender = self.get_unit_count(Player.Defender, UnitType.Tech)
            fp_defender = self.get_unit_count(Player.Defender, UnitType.Firewall)
            pp_defender = self.get_unit_count(Player.Defender, UnitType.Program)
            ai_defender = self.get_unit_count(Player.Defender, UnitType.AI)

            heuristic_value = (
                    ((3 * vp_attacker) + (3 * tp_attacker) + (3 * fp_attacker) + (3 * pp_attacker) + (
                            9999 * ai_attacker)) -
                    ((3 * vp_defender) + (3 * tp_defender) + (3 * fp_defender) + (3 * pp_defender) + (
                            9999 * ai_defender))
            )
        elif heuristic_type == 'e1':
            aggregate_health_amount = self.get_aggregate_health(Player.Attacker) - self.get_aggregate_health(
                Player.Defender)
            potential_damage_delta = self.get_potential_damage_delta()
            heuristic_value = aggregate_health_amount + potential_damage_delta

        return int(heuristic_value)

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool, start_time: datetime,
                max_time: float) -> Tuple[int, CoordPair | None]:
        """minimax algorithm"""
        is_game_clone = True
        current_elapsed_time = (datetime.now() - start_time).total_seconds()

        # If depth has been reached or the game is finished or max time has approached, return a heuristic value
        if self.is_finished():
            if self.has_winner() == Player.Attacker:
                return MAX_HEURISTIC_SCORE, None
            elif self.has_winner() == Player.Defender:
                return MIN_HEURISTIC_SCORE, None
        elif depth == 0 or current_elapsed_time > max_time:
            return self.evaluate('e1'), None

        best_move = None
        if maximizing_player:  # for the maximizing player
            max_eval = float('-inf')
            for move in self.move_candidates():
                game_clone = self.clone()
                (success, result) = game_clone.perform_move(move, is_game_clone)
                if success:
                    eval_value, _ = game_clone.minimax(depth - 1, alpha, beta, False, start_time, max_time)
                else:
                    continue

                if eval_value > max_eval:
                    max_eval = eval_value
                    best_move = move

                if max_eval > beta:
                    break

                alpha = max(alpha, max_eval)

            return max_eval, best_move

        else:  # for the minimizing player
            min_eval = float('inf')
            for move in self.move_candidates():
                game_clone = self.clone()
                (success, result) = game_clone.perform_move(move, is_game_clone)
                if success:
                    eval_value, _ = game_clone.minimax(depth - 1, alpha, beta, True, start_time, max_time)
                else:
                    continue

                if eval_value < min_eval:
                    min_eval = eval_value
                    best_move = move

                if min_eval < alpha:
                    break

                beta = min(beta, min_eval)

            return min_eval, best_move

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax OR alpha-beta pruning."""
        max_time = self.options.max_time
        start_time = datetime.now()
        max_depth = int(self.options.max_depth)
        if self.options.alpha_beta:
            print("CALL ALPHA-BETA")
        else:
            score, move = self.minimax(max_depth, float('-inf'), float('inf'), self.next_player == Player.Attacker,
                                       start_time, max_time)  # Attacker will always be the initial maximizer

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds

        print(f"Heuristic score: {score}")
        self.write_output(f"Heuristic score: {score}\n")
        print(f"Evals per depth: ", end='')
        self.write_output(f"Evals per depth: ")
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
            self.write_output(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s\n")

        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        self.write_output(f"Elapsed time: {elapsed_seconds:0.1f}s\n")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None


##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    # set up game options
    options = Options()

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    try:
        with open('options.json', 'r') as f:
            print("Loading options from options.json")
            config_data = json.load(f)
            options.dim = config_data.get('dim', options.dim)
            options.max_depth = config_data.get('max_depth', options.max_depth)
            options.min_depth = config_data.get('min_depth', options.min_depth)
            options.max_time = config_data.get('max_time', options.max_time)
            options.game_type = GameType.get_game_type(config_data.get('game_type', args.game_type))
            options.alpha_beta = config_data.get('alpha_beta', options.alpha_beta)
            options.max_turns = config_data.get('max_turns', options.max_turns)
            options.randomize_moves = config_data.get('randomize_moves', options.randomize_moves)
            options.broker = config_data.get('broker', options.broker)
    except FileNotFoundError:
        print("No options.json file found, using defaults.")
        pass

    # create a new game
    game = Game(options=options)

    log_file_path = f'game_trace-{options.alpha_beta}-{options.max_time}-{options.max_turns}.txt'
    game.start_logging(log_file_path)
    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            game.write_output(f"{winner.name} wins in {game.turns_played} turns\n")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

    game.stop_logging()


##############################################################################################################

if __name__ == '__main__':
    main()
