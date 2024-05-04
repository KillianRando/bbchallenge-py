import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tabulate import tabulate
import tqdm


class TMConfiguration(object):
    def __init__(self, pos, state, step, tape, tape_length, head_direction):
        """
            Args:
                pos: position of the head
                state: current state
                step: current step
                tape: tape
                tape_length: length of the tape
                head_direction: direction of the head (`>`, or `<`)
        """
        self.pos = pos
        self.state = state
        self.step = step
        self.tape = tape
        self.tape_length = tape_length
        self.head_direction = head_direction

    def __str__(self):
        return f"step={self.step}, pos={self.pos}, state={ithl(self.state)},"+\
        f" tape_len={self.tape_length} tape=`{tape_to_str(self.tape, self.pos, self.head_direction)}`"


db_path = "C:/Users/rando/Documents/Master 1/S2/bb-challenge/bbchallenge/Ressources/TMDB"


# nb de machind dans la base
n = os.path.getsize("C:/Users/rando/Documents/Master 1/S2/bb-challenge/bbchallenge/Ressources/TMDB")
# print((n)/30-1)




def get_machine_i(machine_db_path, i, db_has_header=True):
    with open(machine_db_path, "rb") as f:
        c = 1 if db_has_header else 0
        f.seek(30*(i+c))
        return f.read(30)


def ithl(i):
    return chr(ord("A") + i)


def g(move):
    if move == 0:
        return "R"
    return "L"


def pptm(machine, return_repr=False):
    headers = ["s", "0", "1"]
    table = []

    for i in range(5):
        row = [ithl(i)]
        for j in range(2):
            write = machine[6 * i + 3 * j]
            move = machine[6 * i + 3 * j + 1]
            goto = machine[6 * i + 3 * j + 2] - 1

            if goto == -1:
                row.append("???")
                continue

            row.append(f"{write}{g(move)}{ithl(goto)}")
        table.append(row)

    # if not return_repr:
        # print(tabulate(table, headers=headers))
    # else:
        return tabulate(table, headers=headers)




def step(machine, curr_state, curr_pos, tape):
    if not curr_pos in tape:
        tape[curr_pos] = 0

    write = machine[curr_state * 6 + 3 * tape[curr_pos]]
    move = machine[curr_state * 6 + 3 * tape[curr_pos] + 1]
    goto = machine[curr_state * 6 + 3 * tape[curr_pos] + 2] - 1

    if goto == -1:
        return None, None

    tape[curr_pos] = write
    next_pos = curr_pos + (-1 if move else 1)
    return goto, next_pos


def simulate(machine, time_limit=1000, mini=-10, maxi=-10):
    curr_time = 0
    curr_state = 0
    curr_pos = 0
    tape = {}

    config = TMConfiguration(curr_pos, curr_state, curr_time, tape, len(tape), '>')

    while curr_state != None and curr_time < time_limit:
        next_state, next_pos = step(machine, curr_state, curr_pos, tape)
        if curr_state is not None:
             config = TMConfiguration(next_pos, next_state, curr_time, tape, len(tape), '>' if next_pos - curr_pos > 0 else '<')
             print(config)
        else:
            print("HALT")
        curr_time += 1
        curr_state = next_state
        curr_pos = next_pos




def tm_trace_to_image(machine, width=900, height=1000, origin=0.5, show_head_direction=False):
    img = Image.new('RGB', (width, height), color='black')
    pixels = img.load()

    tape = {}
    curr_time = 0
    curr_state = 0
    curr_pos = 0
    tape = {}

    for row in range(1, height):
        last_pos = curr_pos
        curr_state, curr_pos = step(machine, curr_state, curr_pos, tape)

        if curr_state is None:  # halt
            return img

        for col in range(width):
            pos = col - width * (origin)

            if pos in tape:
                pixels[col, row] = (255, 255, 255) if tape[pos] == 1 else (0, 0, 0)
                # pixels[col,row-1] = colors[curr_state-1]

            if pos == curr_pos and show_head_direction:
                pixels[col, row] = (255, 0, 0) if curr_pos > last_pos else (0, 255, 0)

                # img = zoom_at(img,*zoom)
    return img


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


def repr_to_bytes(rep):
    to_rep = bytearray()
    for a in rep:
        to_rep.append(a)
    return to_rep


def tape_to_str(tape, pos, tape_head):
    min_pos = min(tape.keys())
    max_pos = max(tape.keys())
    s = ""
    for i in range(min_pos,max_pos+1):
        if i == pos and tape_head == '>':
            s += ">"
        s += "." if tape[i] == 0 else "#"
        if i == pos and tape_head == '<':
            s += "<"
    return s


def get_min_pos_in_tape(tape):
    val = 100
    for i in tape.keys():
        if i < val:
            val = i
    return val


def cmp_tape(curr_pos, max_pos, tape1, tape2):
    i = max_pos
    if len(tape1) != len(tape2):
        return False
    print("\n")
    while i != curr_pos - 1:
        print(tape1[i], "  ", tape2[i])
        if tape1[i] != tape2[i]: return False
        i -= 1

    return True


def config_copy(config):
    return TMConfiguration(config.pos, config.state, config.step, config.tape.copy(), config.tape_length, config.head_direction)




################################################################################################################


def get_t(machine, n):
    curr_time = 0
    curr_state = 0
    curr_pos = 0
    tape = {}
    max_pos = -1000000
    config = TMConfiguration(curr_pos, curr_state, curr_time, tape, len(tape), '>')

    # Simulation of the machine on n steps
    for i in range(n):
        curr_time += 1
        curr_state, next_pos = step(machine, curr_state, curr_pos, tape)
        older_pos = curr_pos
        curr_pos = next_pos
        config = TMConfiguration(curr_pos, curr_state, curr_time, tape, len(tape), '>' if curr_pos - older_pos > 0 else '<')

        print("\n", config)
        print(tape)
        print("boucle 1")

    min_pos = get_min_pos_in_tape(tape)
    print(min_pos)

    # Find first state of an erase transition
    while curr_pos > min_pos - 1:
        curr_time += 1
        curr_state, next_pos = step(machine, curr_state, curr_pos, tape)
        older_pos = curr_pos
        curr_pos = next_pos
        config = TMConfiguration(curr_pos, curr_state, curr_time, tape, len(tape), '>' if curr_pos - older_pos > 0 else '<')

        print("\n", config)
        print(tape)
        print("boucle 2")

    record_config = [config_copy(config)]

    # Enter a rebound config
    while curr_pos > max_pos:
        if max_pos < curr_pos:
            max_pos = curr_pos

        curr_time += 1
        curr_state, next_pos = step(machine, curr_state, curr_pos, tape)
        older_pos = curr_pos
        curr_pos = next_pos
        config = TMConfiguration(curr_pos, curr_state, curr_time, tape, len(tape), '>' if curr_pos - older_pos > 0 else '<')
        record_config.append(config_copy(config))

        # print("\n", config)
        # print(tape)
        # print("boucle 3")

    record_config.pop(len(record_config) - 1)
    tape = record_config[len(record_config) - 1].tape
    curr_pos = record_config[len(record_config) - 1].pos
    curr_state = record_config[len(record_config) - 1].state
    curr_time = record_config[len(record_config) - 1].step
    record_config.pop(len(record_config) - 1)

    final_record = None
    find_t = False

    while curr_state > min_pos and not find_t:
        curr_time += 1
        curr_state, next_pos = step(machine, curr_state, curr_pos, tape)
        older_pos = curr_pos
        curr_pos = next_pos
        config = TMConfiguration(curr_pos, curr_state, curr_time, tape, len(tape), '>' if curr_pos - older_pos > 0 else '<')
        print("\n")
        print(config)
        print("boucle 4")

        for record in reversed(record_config):
            print(record)
            print("boucle 4 record")
            if record.pos > config.pos:
                if cmp_tape(curr_pos + 1, max_pos, tape, record.tape):
                    if record.head_direction == '>' and config.head_direction == '<':
                        find_t = True
                        print("\n", record)
                        print(config)
                        break



R, L = 0, 1
counter = get_machine_i(db_path, 11004366)

get_t(counter, 50)



