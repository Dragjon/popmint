import argparse
import subprocess
import time
import Popitto.Framework as pt
import os
import random
import math
from dataclasses import dataclass
from GUI import *

RED = "\033[0;31m"
GREEN = "\033[0;32m"
BROWN = "\033[0;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
LIGHT_GRAY = "\033[0;37m"
LIGHT_RED = "\033[1;31m"
LIGHT_GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
LIGHT_BLUE = "\033[1;34m"
LIGHT_PURPLE = "\033[1;35m"
LIGHT_CYAN = "\033[1;36m"
LIGHT_WHITE = "\033[1;37m"
END = "\033[0m"

def run_engine(cmd, name):
    # Launch the engine process
    engine = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    info_handler = engine.stdout
    engine.stdin.write("upi\n")
    engine.stdin.flush()
    engine.stdin.write(f"upinewgame\nisready\n")
    engine.stdin.flush()
    # Confirm engine is ready
    while True:
        line = info_handler.readline().strip()
        if line == "readyok":
            break
    return engine, info_handler

'''
Normal elo functions
Taken from: https://github.com/jw1912/SPRT/blob/main/sprt.py
'''

@dataclass
class Probability:
    win: float
    loss: float
    draw: float


@dataclass
class BayesElo:
    elo: float
    draw: float


def expected_score(x: float) -> float:
    return 1.0 / (1.0 + pow(10, -x / 400.0))


def erf_inv(x):
    a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
    y = math.log(1 - x * x)
    z = 2 / (math.pi * a) + y / 2
    return math.copysign(math.sqrt(math.sqrt(z * z - y / a) - z), x)


def phi_inv(p):
    return math.sqrt(2)*erf_inv(2*p-1)

def elo(score: float) -> float:
    if score <= 0 or score >= 1:
        return 0.0
    return -400 * math.log10(1 / score - 1)


'''
Pentanomial SPRTS
Taken from: https://github.com/raklaptudirm/arbiter/blob/master/pkg/eve/stats/penta.go
'''

def nEloToScore(nelo, r):
	return nelo*math.sqrt(2)*r/(800/math.log(10)) + 0.5

def penta_llr(lls, lds, dds, wds, wws, elo0, elo1):
	N = lls+lds+dds+wds+wws + 2.5

	ll = (lls + 0.5) / N # measured loss-loss probability
	ld = (lds + 0.5) / N # measured loss-draw probability
	dd = (dds + 0.5) / N # measured win-loss/draw-draw probability
	wd = (wds + 0.5) / N # measured win-draw probability
	ww = (wws + 0.5) / N # measured win-win probability

	# empirical mean of random variable
	mu = ww + 0.75*wd + 0.5*dd + 0.25*ld

	# standard deviation (multiplied by sqrt of N) of the random variable
	r = math.sqrt(
		ww*math.pow(1-mu, 2) +
        wd*math.pow(0.75-mu, 2) +
        dd*math.pow(0.50-mu, 2) +
        ld*math.pow(0.25-mu, 2) +
        ll*math.pow(0.00-mu, 2),
	)

	# convert elo bounds to score
	mu0 = nEloToScore(elo0, r)
	mu1 = nEloToScore(elo1, r)

	# deviation to the score bounds
	r0 = ww*math.pow(1-mu0, 2) + \
		wd*math.pow(0.75-mu0, 2) + \
		dd*math.pow(0.50-mu0, 2) + \
		ld*math.pow(0.25-mu0, 2) + \
		ll*math.pow(0.00-mu0, 2)
	r1 = ww*math.pow(1-mu1, 2) + \
		wd*math.pow(0.75-mu1, 2) + \
		dd*math.pow(0.50-mu1, 2) + \
		ld*math.pow(0.25-mu1, 2) + \
		ll*math.pow(0.00-mu1, 2)

	if r0 == 0 or r1 == 0 :
		return 0

	# log-likelihood ratio (llr)
	# note: this is not the exact llr formula but rather a simplified yet
	# very accurate approximation. see http://hardy.uhasselt.be/Fishtest/support_MLE_multinomial.pdf
	return 0.5 * N * math.log(r0/r1)

def penta_elo(lls, lds, dds, wds, wws):
	N = lls+lds+dds+wds+wws + 2.5 # total number of pairs

	ll = (lls + 0.5) / N # measured loss-loss probability
	ld = (lds + 0.5) / N # measured loss-draw probability
	dd = (dds + 0.5) / N # measured win-loss/draw-draw probability
	wd = (wds + 0.5) / N # measured win-draw probability
	ww = (wws + 0.5) / N # measured win-win probability

	# empirical mean of random variable
	mu = ww + 0.75*wd + 0.5*dd + 0.25*ld

	# standard deviation of the random variable
	sigma = math.sqrt(
		ww*math.pow(1-mu, 2)+
        wd*math.pow(0.75-mu, 2)+
        dd*math.pow(0.50-mu, 2)+
        ld*math.pow(0.25-mu, 2)+
        ll*math.pow(0.00-mu, 2),
	) / math.sqrt(N)

	muMax = mu + phi_inv(0.025)*sigma # upper bound
	muMin = mu + phi_inv(0.975)*sigma # lower bound

	return elo(muMin), elo(mu), elo(muMax)


def flatten_2d_array(board):
    flattened_list = [str(element) for row in board.board for element in row]
    return ''.join(flattened_list) + str(board.turn)

def run_match(engine1_cmd, engine2_cmd, engine1_name, engine2_name, openings_file, elo_interval, time_control, rounds, pgn_out, repeat_rounds=False, debug=False, sprt=False, elo0 = 0, elo1 = 5, alpha = 0.05, beta = 0.05):
    # Initialize engines
    engine1, info1 = run_engine(engine1_cmd, "Engine1")
    engine2, info2 = run_engine(engine2_cmd, "Engine2")
    
    # Load openings from TXT file
    openings = []
    if openings_file:
        with open(openings_file, 'r') as f:
            openings = f.readlines()

    # Time control
    base_time, increment = time_control.split('+')
    base_time = float(base_time)
    increment = float(increment)

    # Elo for engine 1
    wdl = [0, 0, 0]
    pntnml = [0, 0, 0, 0, 0] # ll, ld, dd/wl, wd, ww  

    print(f"{CYAN}Starting match between {engine1_name} and {engine2_name}{END}")

    total_games = 0

    # Match loop
    for round_num in range(1, rounds + 1):
        finish_sprt = False
        random.shuffle(openings)
        penta_score = [0, 0]
        for game_num in range(1, 3 if repeat_rounds else 2):
            total_games += 1
            if game_num != 2: 
                print(f"{LIGHT_WHITE}Starting Game{END} {total_games} | {engine1_name} vs {engine2_name}")
            else:
                print(f"{LIGHT_GRAY}Starting Game{END} {total_games} | {engine2_name} vs {engine1_name}")

            # Select opening randomly if provided
            board = pt.PopIt()
            if openings_file:
                tmpArray, turn = pt.stringToArray(openings[0].strip())
                board = pt.PopIt(tmpArray, turn)

            chosen_opening = openings[0].strip() if openings_file else "0000000000000000000000000000000000001"

            # Reset clocks for the game
            engine1_time = base_time
            engine2_time = base_time

            result = None
            pgn_moves = []

            while True:
                print_board(board.board)
                start_time = time.time()
                wasted_time = time.time() - start_time
                if board.turn == pt.FIRST:
                    if game_num == 2:
                        current_engine = engine2
                        current_time = engine2_time
                    else:
                        current_engine = engine1
                        current_time = engine1_time
                else:
                    if game_num == 2:
                        current_engine = engine1
                        current_time = engine1_time
                    else:
                        current_engine = engine2
                        current_time = engine2_time

                current_engine.stdin.write(f"position {flatten_2d_array(board)}\ngo time1 {int((engine1_time if game_num != 2 else engine2_time) * 1000)} time2 {int((engine2_time if game_num != 2 else engine1_time)* 1000)} inc1 {int(increment * 1000)} inc2 {int(increment * 1000)}\n")

                if debug:
                    print(f"position {flatten_2d_array(board)}\ngo time1 {int((engine1_time if game_num != 2 else engine2_time) * 1000)} time2 {int((engine2_time if game_num != 2 else engine1_time)* 1000)} inc1 {int(increment * 1000)} inc2 {int(increment * 1000)}\n")

                current_engine.stdin.flush()
                best_row = None
                best_pops = None
                best_move = (None, None)
                while True:
                    line = current_engine.stdout.readline().strip()
                    if line.startswith("bestmove"):
                        best_row = int(line.split()[1])
                        best_pops = int(line.split()[2])
                        best_move = (best_row, best_pops)
                        break

                # Make the move
                if best_row == None or best_pops == None or best_row > 5 or best_pops > 6 or pt.moveGen(board)[best_row] == 0:
                    if board.turn == pt.FIRST:
                        print(f"Finished Game {total_games} | {RED}{engine2_name if game_num != 2 else engine1_name} Wins due to illegal move [0-1]{END}")
                        result = "0-1"  # Player 2 wins due to illegal move or other errors
                    else:
                        print(f"Finished Game {total_games} | {RED}{engine1_name if game_num != 2 else engine2_name} Wins due to illegal move [1-0]{END}")
                        result = "1-0"  # Player 1 wins due to illegal move or other errors
                    break
                
                pgn_moves.append(best_move)
                board = board.makeMove(best_row, best_pops)
                if debug:
                    pt.printPopIt(board)

                # Check for end of game conditions
                if pt.isCheckMate(board):
                    if board.turn == pt.FIRST:
                        print(f"Finished Game {total_games} | {LIGHT_GREEN}{engine2_name if game_num != 2 else engine1_name} Wins by checkmate [0-1]{END}")
                        result = "0-1"  # Player 2 wins by checkmate
                    else:
                        print(f"Finished Game {total_games} | {LIGHT_GREEN}{engine1_name if game_num != 2 else engine2_name} Wins by checkmate [1-0]{END}")
                        result = "1-0"  # Player 1 wins by checkmate
                    break

                # Adjust time for the next move
                current_time += increment
                current_time -= (time.time() - start_time)

                if current_time < 0:
                    if board.turn == pt.FIRST:
                        print(f"Finished Game {total_games} | {LIGHT_RED}{engine1_name if game_num != 2 else engine2_name} Wins due to timeout [1-0]{END}")
                        result = "1-0"  # Player 1 wins on time because Black has < 0 time after making move, switching the turns
                    else:
                        print(f"Finished Game {total_games} | {LIGHT_RED}{engine2_name if game_num != 2 else engine1_name} Wins due to timeout [0-1]{END}")
                        result = "0-1"  # Player 2 wins on time
                    break

                # Update time remaining for each player
                if board.turn == pt.FIRST:
                    if game_num != 2:
                        engine1_time = current_time
                    else:
                        engine2_time = current_time
                else:
                    if game_num == 2:
                        engine1_time = current_time
                    else:
                        engine2_time = current_time

            if game_num != 2:
                if result == "1-0":
                    penta_score[0] = 1
                    wdl[0] += 1

                else:
                    penta_score[0] = 0.5
                    wdl[2] += 1

            else:
                if result == "1-0":
                    penta_score[1] = 0.5
                    wdl[2] += 1
                else:
                    penta_score[1] = 1
                    wdl[0] += 1
                
                # ww
                if penta_score[1] == 1 and penta_score[0] == 1:
                    pntnml[4] += 1

                # wl
                elif (penta_score[1] == 1 and penta_score[0] == 0.5) or (penta_score[1] == 0.5 and penta_score[0] == 1):
                    pntnml[2] += 1

                # ll
                elif penta_score[1] == 0.5 and penta_score[0] == 0.5:
                    pntnml[0] += 1
            '''
            elo_min, elo_std, elo_max = elo_wld(wdl[0], wdl[1], wdl[2])
            llr = normal_llr(wdl[0], wdl[2], wdl[1], elo0, elo1)
            '''
            llrpenta = penta_llr(pntnml[0], pntnml[1], pntnml[2], pntnml[3], pntnml[4], elo0, elo1)
            elopenta_min, elopenta_std, elopenta_max = penta_elo(pntnml[0], pntnml[1], pntnml[2], pntnml[3], pntnml[4])

            lower = math.log(args.beta / (1 - args.alpha))
            upper = math.log((1 - args.beta) / args.alpha)

            h1true = llrpenta >= upper
            h0true = llrpenta <= lower
            if sprt:
                if h1true or h0true:
                    message = "H0 is accepted"
                    if h1true:
                        message = "H1 is accepted"
                    print(f'''
*****************************************************
* Final Results of {engine1_name} vs {engine2_name}
*****************************************************
* TC     | {BLUE}{base_time}+{increment}{END}
* Params | {GREEN}Elo0={elo0}, Elo1={elo1}, Alpha={alpha}, Beta={beta}{END}
* Games  | {LIGHT_CYAN}{total_games}{END}
* WDL    | {LIGHT_GREEN}{wdl[0]}{END} | {LIGHT_GRAY}{wdl[1]}{END} | {LIGHT_RED}{wdl[2]}{END}
* Pntnml | {LIGHT_RED}{pntnml[0]}{END} | {YELLOW}{pntnml[1]}{END} | {LIGHT_GRAY}{pntnml[2]}{END} | {LIGHT_BLUE}{pntnml[3]}{END} | {LIGHT_GREEN}{pntnml[4]}{END}
* Points | {YELLOW}{wdl[0] + 0.5*wdl[1]} ({(((wdl[0] + 0.5*wdl[1]) / (wdl[0] + wdl[1] + wdl[2])) * 100):.3}%){END}
* Draw R.| {LIGHT_WHITE}{(wdl[1]/(wdl[0]+wdl[2])):.3}{END}
* Elo    | {BROWN}{elopenta_std:.3f}{END} +- {(abs(elopenta_max - elopenta_min) / 2):.3f} [{elopenta_min:.3f}, {elopenta_max:.3f}]
* LLR    | {LIGHT_PURPLE}{llrpenta:.3}{END} [{elo0}, {elo1}] ({lower:.3}, {upper:.3})
* Conc.  | {message}
*****************************************************
                    ''')
                    finish_sprt = True

                
            if total_games % elo_interval == 0:
                print(f'''
*****************************************************
* Partial results of {engine1_name} vs {engine2_name}
*****************************************************
* TC     | {BLUE}{base_time}+{increment}{END}
* Params | {GREEN}Elo0={elo0}, Elo1={elo1}, Alpha={alpha}, Beta={beta}{END}
* Games  | {LIGHT_CYAN}{total_games}{END}
* WDL    | {LIGHT_GREEN}{wdl[0]}{END} | {LIGHT_GRAY}{wdl[1]}{END} | {LIGHT_RED}{wdl[2]}{END}
* Pntnml | {LIGHT_RED}{pntnml[0]}{END} | {YELLOW}{pntnml[1]}{END} | {LIGHT_GRAY}{pntnml[2]}{END} | {LIGHT_BLUE}{pntnml[3]}{END} | {LIGHT_GREEN}{pntnml[4]}{END}
* Points | {YELLOW}{wdl[0] + 0.5*wdl[1]} ({(((wdl[0] + 0.5*wdl[1]) / (wdl[0] + wdl[1] + wdl[2])) * 100):.3}%){END}
* Draw R.| {LIGHT_WHITE}{(wdl[1]/(wdl[0]+wdl[2])):.3}{END}
* Elo    | {BROWN}{elopenta_std:.3f}{END} +- {(abs(elopenta_max - elopenta_min) / 2):.3f} [{elopenta_min:.3f}, {elopenta_max:.3f}]
* LLR    | {LIGHT_PURPLE}{llrpenta:.3}{END} [{elo0}, {elo1}] ({lower:.3}, {upper:.3})
*****************************************************
                ''')

            # Write PGN to file
            pgn_file = pgn_out if pgn_out else "engine_match.pgn"
            file_mode = "a" if os.path.exists(pgn_file) else "w"

            with open(pgn_file, file_mode) as f:
                # If the file didn't exist before, write header information
                f.write(f'[Event "SPRT Popmint"]\n')
                f.write(f'[Site "Desktop"]\n')
                f.write(f'[Date "{time.strftime("%Y.%m.%d")}"]\n')
                f.write(f'[Round "{round_num}-{game_num}"]\n')
                f.write(f'[White "{engine1_name if game_num != 2 else engine2_name}"]\n')
                f.write(f'[Black "{engine1_name if game_num == 2 else engine2_name}"]\n')
                f.write(f'[Result "{result}"]\n')
                f.write(f'[Pos "{chosen_opening}"]\n')

                i = 1


                for move in pgn_moves:
                    move_row, num_pops = move
                    if i % 2 != 0:
                        f.write(f"{math.ceil(i/2)}. ({move_row}, {num_pops}) ")
                    else:
                        f.write(f"({move_row}, {num_pops}) ")

                    i += 1

                f.write("\n\n")

            if finish_sprt:
                break
        if finish_sprt:
            break

    # Close engines
    engine1.kill()
    engine2.kill()

    print(f"Match finished. PGN saved to {pgn_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engine match manager for UPI engines using Popitto")
    parser.add_argument("-engine1", "--engine1_cmd", type=str, required=True, help="Engine 1 path")
    parser.add_argument("-name1", "--engine1_name", type=str, required=True, help="Name of Engine 1")
    parser.add_argument("-engine2", "--engine2_cmd", type=str, required=True, help="Engine 2 path")
    parser.add_argument("-name2", "--engine2_name", type=str, required=True, help="Name of Engine 2")
    parser.add_argument("-openings", "--openings_file", type=str, default=None, help="File containing openings in EPD format")
    parser.add_argument("-interval", "--elo_interval", type=int, default=20, help="Interval to print elo")
    parser.add_argument("-tc", "--time_control", type=str, required=True, help="Time control in format base+increment (e.g., 8+8000 for 8 seconds and 80 milliseconds)")
    parser.add_argument("-rd", "--rounds", type=int, default=1, help="Number of rounds to play")
    parser.add_argument("-pgn", "--pgn_out", type=str, default=None, help="Output PGN file path")
    parser.add_argument("-rp", "--repeat", action="store_true", help="Repeat rounds with colors switched for the second game")
    parser.add_argument("-db", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-sprt", "--sprt", action="store_true", help="Enable sprt mode")
    parser.add_argument("-gui", "--gui", action="store_true", help="Enable GUI")
    parser.add_argument("-e0", "--elo0", type=int, default=0, help="True elo < elo0 with alpha = 0.05 means a 0.05 chance of passing")
    parser.add_argument("-e1", "--elo1", type=int, default=5, help="True elo > elo1 with beta = 0.05 means a 0.95 chance of passing")
    parser.add_argument("-a", "--alpha", type=float, default=0.05, help="True elo < elo1 with alphs = 0.05 means a 0.05 chance of passing")
    parser.add_argument("-b", "--beta", type=float, default=0.05, help="True elo > elo1 with beta = 0.05 means a 0.95 chance of passing")

    args = parser.parse_args()

    run_match(args.engine1_cmd, args.engine2_cmd, args.engine1_name, args.engine2_name, args.openings_file, args.elo_interval, args.time_control, args.rounds, args.pgn_out, args.repeat, args.debug, args.sprt, args.elo0, args.elo1, args.alpha, args.beta)
