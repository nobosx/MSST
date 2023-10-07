import numpy as np
import speck as sp
from time import time
from os import urandom, path, makedirs
from util import extract_selected_bits_int, extract_bits_and_combine, combine_ints, cal_hw

# Try to determine the borrow bit b[kg_low] at bit position kg_low using m consecutive bits [kg_low-1~kg_low-m]
# Return: b[kg_low], is_valid
# is_valid: whether the borrow bit at bit position kg_low of a ciphertext can be recovered
def determine_borrow_bit_from_low(z, y, kg_low, m):
    mask_val = (1 << m) - 1
    a = (z >> (kg_low - m)) & mask_val
    b = (y >> (kg_low - m)) & mask_val
    return a < b, a != b
    
def expand_key_guess_space(key_guess_bits):
    kg = np.zeros((1,), dtype=np.uint32)
    for kg_bit in key_guess_bits:
        mask = 1 << kg_bit
        kg = np.concatenate((kg, kg ^ mask))
    assert kg[0] == 0
    return kg

# Generate a random user key for a key recovery attack
def gen_user_key(nr, master_key_bit_length=96):
    WORD_SIZE = sp.WORD_SIZE()
    assert master_key_bit_length % WORD_SIZE == 0
    m = master_key_bit_length // WORD_SIZE
    assert m == 3 or m == 4
    key = np.frombuffer(urandom(m * 4), dtype=np.uint32).reshape(m, 1)
    ks = sp.expand_key(key, nr)
    return ks

def make_plaintext_structure(diff, neutral_bits, given_plaintext=None):
    WORD_SIZE = sp.WORD_SIZE()
    MASK_VAL = 2**WORD_SIZE - 1
    if given_plaintext is None:
        p0l = np.frombuffer(urandom(4), dtype=np.uint32)
        p0r = np.frombuffer(urandom(4), dtype=np.uint32)
    else:
        p0l = given_plaintext[0].copy().reshape(-1, 1)
        p0r = given_plaintext[1].copy().reshape(-1, 1)
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0
        d1 = 0
        for j in i:
            d = 1 << j
            d0 |= d >> WORD_SIZE
            d1 |= d & MASK_VAL
        p0l = np.concatenate([p0l, p0l ^ d0], axis=-1)
        p0r = np.concatenate([p0r, p0r ^ d1], axis=-1)
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r

def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r

# Execute the core key guessing and filtering pipeline
# The filtering process ueses a lookup table as a distingusiher
# The total key guessing space is divided into three stages
# stage1: guess the bits of the subkey sk1 that are related to the borrow bit in the modular substraction operation of the last round
# stage2: guess the bits of the subkeys sk0 and sk1 that are related to the borrow bit Modular substraction operation of the penultimate round
# stage3: guess other bits of the subkeys sk0 and sk1 that are related to the input of the distinguisher
# sk0: the subkey of the penultimate round
# sk1: the subkey of the last round
def key_guess_two_rounds(cipher_structure, lookup_table, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, total_kg_bits_per_round, borrow_pos_per_round, m_per_round):
    c0l, c0r, c1l, c1r = cipher_structure
    structure_size = len(c0l)
    c0r = c0l ^ c0r; c1r = c1l ^ c1r
    y0 = sp.ror(c0r, sp.BETA()); y1 = sp.ror(c1r, sp.BETA())
    borrow_kg1_space_stage1 = expand_key_guess_space(borrow_kg_bits_stage1)
    borrow_kg1_space_stage2 = expand_key_guess_space(borrow_kg_bits_per_round_stage2[1])
    borrow_kg0_space_stage2 = expand_key_guess_space(borrow_kg_bits_per_round_stage2[0])
    direct_kg1_space = expand_key_guess_space(direct_kg_bits_per_round[1])
    direct_kg0_space = expand_key_guess_space(direct_kg_bits_per_round[0])
    joint_direct_kg_space_1 = np.tile(direct_kg1_space, len(direct_kg0_space)); joint_direct_kg_space_0 = np.repeat(direct_kg0_space, len(direct_kg1_space))
    direct_kg_size = 2**(len(direct_kg_bits_per_round[0]) + len(direct_kg_bits_per_round[1]))
    total_kg_length = len(borrow_kg_bits_per_round_stage2[0]) + len(direct_kg_bits_per_round[0]) + len(borrow_kg_bits_stage1) + len(borrow_kg_bits_per_round_stage2[1]) + len(direct_kg_bits_per_round[1])
    print("Total kg length:", total_kg_length)
    kg_scores = np.zeros(2**total_kg_length)
    # stage1: guess sk1[borrow_kg_bits_stage1]
    for borrow_kg1_high in borrow_kg1_space_stage1:
        partial_z0_stage1 = c0l ^ borrow_kg1_high; partial_z1_stage1 = c1l ^ borrow_kg1_high
        # Find ciphertext pairs whose borrow bit can be recovered
        _, is_valid0 = determine_borrow_bit_from_low(partial_z0_stage1, y0, borrow_pos_per_round[1], m_per_round[1])
        _, is_valid1 = determine_borrow_bit_from_low(partial_z1_stage1, y1, borrow_pos_per_round[1], m_per_round[1])
        valid_pair_pos = is_valid0 & is_valid1
        partial_z0_stage1 = partial_z0_stage1[valid_pair_pos]; partial_z1_stage1 = partial_z1_stage1[valid_pair_pos]
        valid_y0_stage1 = y0[valid_pair_pos]; valid_y1_stage1 = y1[valid_pair_pos]
        # Stage2: guess sk1[borrow_kg_bits_per_round_stage2[1]] and sk0[borrow_kg_bits_per_round_stage2[0]]
        for borrow_kg1_low in borrow_kg1_space_stage2:
            partial_z0_stage2 = partial_z0_stage1 ^ borrow_kg1_low; partial_z1_stage2 = partial_z1_stage1 ^ borrow_kg1_low
            x0 = partial_z0_stage2 - valid_y0_stage1; x1 = partial_z1_stage2 - valid_y1_stage1
            x0 = sp.rol(x0, sp.ALPHA()); x1 = sp.rol(x1, sp.ALPHA())
            valid_v0_stage2 = sp.ror(valid_y0_stage1 ^ x0, sp.BETA()); valid_v1_stage2 = sp.ror(valid_y1_stage1 ^ x1, sp.BETA())
            for borrow_kg0 in borrow_kg0_space_stage2:
                partial_w0_stage2 = x0 ^ borrow_kg0; partial_w1_stage2 = x1 ^ borrow_kg0
                _, is_valid0 = determine_borrow_bit_from_low(partial_w0_stage2, valid_v0_stage2, borrow_pos_per_round[0], m_per_round[0])
                _, is_valid1 = determine_borrow_bit_from_low(partial_w1_stage2, valid_v1_stage2, borrow_pos_per_round[0], m_per_round[0])
                valid_pair_pos = is_valid0 & is_valid1
                # Find ciphertext pairs whose borrow bit can be recovered
                partial_z0_stage2_prime = partial_z0_stage2[valid_pair_pos]; partial_z1_stage2_prime = partial_z1_stage2[valid_pair_pos]
                valid_y0_stage2 = valid_y0_stage1[valid_pair_pos]; valid_y1_stage2 = valid_y1_stage1[valid_pair_pos]
                valid_num = len(partial_z0_stage2_prime)
                # Stage3: guess sk0[direct_kg_bits_per_round[0]] and sk1[direct_kg_bits_per_round[1]]
                extended_joint_direct_kg_space_1 = np.repeat(joint_direct_kg_space_1, valid_num)
                extended_joint_direct_kg_space_0 = np.repeat(joint_direct_kg_space_0, valid_num)
                partial_z0_stage2_prime = np.tile(partial_z0_stage2_prime, direct_kg_size); partial_z1_stage2_prime = np.tile(partial_z1_stage2_prime, direct_kg_size)
                valid_y0_stage2 = np.tile(valid_y0_stage2, direct_kg_size); valid_y1_stage2 = np.tile(valid_y1_stage2, direct_kg_size)
                # Decrypt two rounds
                a0 = partial_z0_stage2_prime ^ extended_joint_direct_kg_space_1
                a1 = partial_z1_stage2_prime ^ extended_joint_direct_kg_space_1
                a0 = sp.rol(a0 - valid_y0_stage2, sp.ALPHA())
                a1 = sp.rol(a1 - valid_y1_stage2, sp.ALPHA())
                b0 = sp.ror(a0 ^ valid_y0_stage2, sp.BETA())
                b1 = sp.ror(a1 ^ valid_y1_stage2, sp.BETA())
                a0 = a0 ^ extended_joint_direct_kg_space_0 ^ borrow_kg0
                a1 = a1 ^ extended_joint_direct_kg_space_0 ^ borrow_kg0
                a0 = sp.rol(a0 - b0, sp.ALPHA())
                a1 = sp.rol(a1 - b1, sp.ALPHA())
                table_input_int = extract_bits_and_combine([a0, b0, a1, b1], selected_bits)
                Z = lookup_table[table_input_int].reshape(direct_kg_size, valid_num)
                Z = np.sum(Z, axis=1)
                Z = Z * structure_size / valid_num
                embedded_kg = combine_ints([extract_selected_bits_int(borrow_kg0 ^ joint_direct_kg_space_0, total_kg_bits_per_round[0], True), extract_selected_bits_int(borrow_kg1_high ^ borrow_kg1_low ^ joint_direct_kg_space_1, total_kg_bits_per_round[1], True)], [len(total_kg_bits_per_round[0]), len(total_kg_bits_per_round[1])])
                kg_scores[embedded_kg] = Z
    return kg_scores

def guess_two_rounds_and_filter(cipher_structure, lookup_table, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, total_kg_bits_per_round, borrow_pos_per_round, m_per_round, c):
    c0l, c0r, c1l, c1r = cipher_structure
    structure_size = len(c0l)
    c0r = c0l ^ c0r; c1r = c1l ^ c1r
    y0 = sp.ror(c0r, sp.BETA()); y1 = sp.ror(c1r, sp.BETA())
    borrow_kg1_space_stage1 = expand_key_guess_space(borrow_kg_bits_stage1)
    borrow_kg1_space_stage2 = expand_key_guess_space(borrow_kg_bits_per_round_stage2[1])
    borrow_kg0_space_stage2 = expand_key_guess_space(borrow_kg_bits_per_round_stage2[0])
    direct_kg1_space = expand_key_guess_space(direct_kg_bits_per_round[1])
    direct_kg0_space = expand_key_guess_space(direct_kg_bits_per_round[0])
    joint_direct_kg_space_1 = np.tile(direct_kg1_space, len(direct_kg0_space)); joint_direct_kg_space_0 = np.repeat(direct_kg0_space, len(direct_kg1_space))
    direct_kg_size = 2**(len(direct_kg_bits_per_round[0]) + len(direct_kg_bits_per_round[1]))
    total_kg_length = len(borrow_kg_bits_per_round_stage2[0]) + len(direct_kg_bits_per_round[0]) + len(borrow_kg_bits_stage1) + len(borrow_kg_bits_per_round_stage2[1]) + len(direct_kg_bits_per_round[1])
    print("Total kg length:", total_kg_length)
    kg_scores = np.zeros(2**total_kg_length)
    # stage1: guess sk1[borrow_kg_bits_stage1]
    for borrow_kg1_high in borrow_kg1_space_stage1:
        partial_z0_stage1 = c0l ^ borrow_kg1_high; partial_z1_stage1 = c1l ^ borrow_kg1_high
        # Find ciphertext pairs whose borrow bit can be recovered
        _, is_valid0 = determine_borrow_bit_from_low(partial_z0_stage1, y0, borrow_pos_per_round[1], m_per_round[1])
        _, is_valid1 = determine_borrow_bit_from_low(partial_z1_stage1, y1, borrow_pos_per_round[1], m_per_round[1])
        valid_pair_pos = is_valid0 & is_valid1
        partial_z0_stage1 = partial_z0_stage1[valid_pair_pos]; partial_z1_stage1 = partial_z1_stage1[valid_pair_pos]
        valid_y0_stage1 = y0[valid_pair_pos]; valid_y1_stage1 = y1[valid_pair_pos]
        # Stage2: guess sk1[borrow_kg_bits_per_round_stage2[1]] and sk0[borrow_kg_bits_per_round_stage2[0]]
        for borrow_kg1_low in borrow_kg1_space_stage2:
            partial_z0_stage2 = partial_z0_stage1 ^ borrow_kg1_low; partial_z1_stage2 = partial_z1_stage1 ^ borrow_kg1_low
            x0 = partial_z0_stage2 - valid_y0_stage1; x1 = partial_z1_stage2 - valid_y1_stage1
            x0 = sp.rol(x0, sp.ALPHA()); x1 = sp.rol(x1, sp.ALPHA())
            valid_v0_stage2 = sp.ror(valid_y0_stage1 ^ x0, sp.BETA()); valid_v1_stage2 = sp.ror(valid_y1_stage1 ^ x1, sp.BETA())
            for borrow_kg0 in borrow_kg0_space_stage2:
                partial_w0_stage2 = x0 ^ borrow_kg0; partial_w1_stage2 = x1 ^ borrow_kg0
                _, is_valid0 = determine_borrow_bit_from_low(partial_w0_stage2, valid_v0_stage2, borrow_pos_per_round[0], m_per_round[0])
                _, is_valid1 = determine_borrow_bit_from_low(partial_w1_stage2, valid_v1_stage2, borrow_pos_per_round[0], m_per_round[0])
                valid_pair_pos = is_valid0 & is_valid1
                # Find ciphertext pairs whose borrow bit can be recovered
                partial_z0_stage2_prime = partial_z0_stage2[valid_pair_pos]; partial_z1_stage2_prime = partial_z1_stage2[valid_pair_pos]
                valid_y0_stage2 = valid_y0_stage1[valid_pair_pos]; valid_y1_stage2 = valid_y1_stage1[valid_pair_pos]
                valid_num = len(partial_z0_stage2_prime)
                # Stage3: guess sk0[direct_kg_bits_per_round[0]] and sk1[direct_kg_bits_per_round[1]]
                extended_joint_direct_kg_space_1 = np.repeat(joint_direct_kg_space_1, valid_num)
                extended_joint_direct_kg_space_0 = np.repeat(joint_direct_kg_space_0, valid_num)
                partial_z0_stage2_prime = np.tile(partial_z0_stage2_prime, direct_kg_size); partial_z1_stage2_prime = np.tile(partial_z1_stage2_prime, direct_kg_size)
                valid_y0_stage2 = np.tile(valid_y0_stage2, direct_kg_size); valid_y1_stage2 = np.tile(valid_y1_stage2, direct_kg_size)
                # Decrypt two rounds
                a0 = partial_z0_stage2_prime ^ extended_joint_direct_kg_space_1
                a1 = partial_z1_stage2_prime ^ extended_joint_direct_kg_space_1
                a0 = sp.rol(a0 - valid_y0_stage2, sp.ALPHA())
                a1 = sp.rol(a1 - valid_y1_stage2, sp.ALPHA())
                b0 = sp.ror(a0 ^ valid_y0_stage2, sp.BETA())
                b1 = sp.ror(a1 ^ valid_y1_stage2, sp.BETA())
                a0 = a0 ^ extended_joint_direct_kg_space_0 ^ borrow_kg0
                a1 = a1 ^ extended_joint_direct_kg_space_0 ^ borrow_kg0
                a0 = sp.rol(a0 - b0, sp.ALPHA())
                a1 = sp.rol(a1 - b1, sp.ALPHA())
                table_input_int = extract_bits_and_combine([a0, b0, a1, b1], selected_bits)
                Z = lookup_table[table_input_int].reshape(direct_kg_size, valid_num)
                Z = np.sum(Z, axis=1)
                Z = Z * structure_size / valid_num
                embedded_kg = combine_ints([extract_selected_bits_int(borrow_kg0 ^ joint_direct_kg_space_0, total_kg_bits_per_round[0], True), extract_selected_bits_int(borrow_kg1_high ^ borrow_kg1_low ^ joint_direct_kg_space_1, total_kg_bits_per_round[1], True)], [len(total_kg_bits_per_round[0]), len(total_kg_bits_per_round[1])])
                kg_scores[embedded_kg] = Z
    # filter
    embedded_kg_space = np.arange(2**total_kg_length, dtype=np.uint32)
    surviving_kgs_scores_list = []
    surviving_kgs = embedded_kg_space[kg_scores > c]
    for surviving_embedded_kg in surviving_kgs:
        surviving_kgs_scores_list.append((surviving_embedded_kg, kg_scores[surviving_embedded_kg]))
    if len(surviving_kgs_scores_list) > 0:
        surviving_kgs_scores_list.sort(key=lambda x: x[1], reverse=True)
    return surviving_kgs_scores_list

# Test the key recovery attack using right plaintext structures
def test_attack(n, structure_size, dis_nr, dis_diff, lookup_table_path, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, borrow_pos_per_round, m_per_round, c, attack_saved_folder):
    lookup_table = np.load(lookup_table_path)
    total_kg_bits_0 = [i for i in reversed(range(sp.WORD_SIZE())) if i in direct_kg_bits_per_round[0] or i in borrow_kg_bits_per_round_stage2[0]]
    total_kg_bits_1 = [i for i in reversed(range(sp.WORD_SIZE())) if i in direct_kg_bits_per_round[1] or i in borrow_kg_bits_stage1 or i in borrow_kg_bits_per_round_stage2[1]]
    total_kg_bits_per_round = [total_kg_bits_0, total_kg_bits_1]
    print("Total kg bits per round:", total_kg_bits_per_round)
    tk_ranks = []
    tk_scores = []
    wk_average_scores = []
    MAX_SUCCEED_KG_DIFF_HW = 2
    sv_whether_succeed = np.zeros(n, dtype=np.uint8)
    sv_whether_surviving_kg = np.zeros(n, dtype=np.uint8)
    sv_attack_time_consumption = np.zeros(n)
    sv_best_kg_diff_hw = []
    sv_num_surviving_kgs = np.zeros(n, dtype=np.uint32)
    sv_best_kg_diff_show_time = np.zeros(1+len(total_kg_bits_0)+len(total_kg_bits_1), dtype=np.uint32)
    for try_index in range(n):
        print("Attack index:", try_index)
        p0l = np.frombuffer(urandom(4*structure_size), dtype=np.uint32)
        p0r = np.frombuffer(urandom(4*structure_size), dtype=np.uint32)
        p1l = p0l ^ dis_diff[0]
        p1r = p0r ^ dis_diff[1]
        ks = gen_user_key(dis_nr + 2)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)
        tk0 = ks[-2][0]; tk1 = ks[-1][0]
        embedded_tk = combine_ints([extract_selected_bits_int(tk0, total_kg_bits_per_round[0], True), extract_selected_bits_int(tk1, total_kg_bits_per_round[1], True)], [len(total_kg_bits_per_round[0]), len(total_kg_bits_per_round[1])])
        start_time = time()
        kg_scores = key_guess_two_rounds((c0l, c0r, c1l, c1r), lookup_table, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, total_kg_bits_per_round, borrow_pos_per_round, m_per_round)
        end_time = time()
        wk_pos = np.ones(2**(len(total_kg_bits_0) + len(total_kg_bits_1)), dtype=bool)
        wk_pos[embedded_tk] = False
        wk_scores = kg_scores[wk_pos]
        tk_score = kg_scores[embedded_tk]
        tk_scores.append(tk_score)
        tk_ranks.append(np.sum(kg_scores > tk_score))
        wk_average_scores.append(np.mean(wk_scores))
        print("True key score: {}, true key rank: {}".format(tk_scores[-1], tk_ranks[-1]))
        print("Running time: {} s".format(end_time - start_time))
        sv_attack_time_consumption[try_index] = end_time - start_time
        surviving_kg_num = np.sum(kg_scores > c)
        if surviving_kg_num > 0:
            sv_whether_surviving_kg[try_index] = 1
            sv_num_surviving_kgs[try_index] = surviving_kg_num
            print("surviving key guess num:", surviving_kg_num)
            best_embedded_kg = np.argmax(kg_scores)
            assert kg_scores[best_embedded_kg] > c
            hw = cal_hw(embedded_tk ^ best_embedded_kg, len(total_kg_bits_0) + len(total_kg_bits_1))
            sv_best_kg_diff_hw.append(hw)
            sv_best_kg_diff_show_time[hw] += 1
            print("Best key guess score:", kg_scores[best_embedded_kg])
            print("Hamming distance between best key guess and true key:", hw)
            if hw <= MAX_SUCCEED_KG_DIFF_HW:
                sv_whether_succeed[try_index] = 1
        else:
            print("No surviving key guess")
        print('')
    print("Average true key score:", np.mean(tk_scores))
    print("Min true key score:", np.min(tk_scores))
    print("Average wrong key scores:", np.mean(wk_average_scores))
    print("Average true key rank:", np.mean(tk_ranks))
    print("Median true key rank:", np.median(tk_ranks))
    print("Max succeed kg diff hw matric:", MAX_SUCCEED_KG_DIFF_HW)
    print("Key surviving time:", np.sum(sv_whether_surviving_kg))
    print("Attack success time:", np.sum(sv_whether_succeed))
    print("Average attack time: {} s".format(np.mean(sv_attack_time_consumption)))
    print("Average surviving key guess number:", np.mean(sv_num_surviving_kgs))
    print("Average diff hw between true key and surviving key guess:", np.mean(sv_best_kg_diff_hw))
    print("Best kg diff hw show time:", sv_best_kg_diff_show_time)
    if not path.isdir(attack_saved_folder):
        makedirs(attack_saved_folder)
    np.save(attack_saved_folder + "tk_scores.npy", np.array(tk_scores))
    np.save(attack_saved_folder + "tk_rank.npy", np.array(tk_ranks))
    np.save(attack_saved_folder + 'sv_whether_surviving_kg.npy', sv_whether_surviving_kg)
    np.save(attack_saved_folder + 'sv_whether_succeed.npy', sv_whether_succeed)
    np.save(attack_saved_folder + 'sv_attack_time.npy', sv_attack_time_consumption)
    np.save(attack_saved_folder + 'sv_best_kg_diff_show_time.npy', sv_best_kg_diff_show_time)
    np.save(attack_saved_folder + 'sv_num_surviving_kgs.npy', sv_num_surviving_kgs)

# Test the key recovery attack using wrong plaintext structures
def test_attack_with_bad_structure(n, structure_size, dis_nr, dis_diff, lookup_table_path, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, borrow_pos_per_round, m_per_round, c, attack_saved_folder):
    lookup_table = np.load(lookup_table_path)
    total_kg_bits_0 = [i for i in reversed(range(sp.WORD_SIZE())) if i in direct_kg_bits_per_round[0] or i in borrow_kg_bits_per_round_stage2[0]]
    total_kg_bits_1 = [i for i in reversed(range(sp.WORD_SIZE())) if i in direct_kg_bits_per_round[1] or i in borrow_kg_bits_stage1 or i in borrow_kg_bits_per_round_stage2[1]]
    total_kg_bits_per_round = [total_kg_bits_0, total_kg_bits_1]
    print("Total kg bits per round:", total_kg_bits_per_round)
    tk_ranks = []
    tk_scores = []
    wk_average_scores = []
    wk_max_score = -10000
    MAX_SUCCEED_KG_DIFF_HW = 2
    sv_whether_surviving_kg = np.zeros(n, dtype=np.uint8)
    sv_attack_time_consumption = np.zeros(n)
    sv_num_surviving_kgs = np.zeros(n, dtype=np.uint32)
    for try_index in range(n):
        print("Attack index:", try_index)
        p0l = np.frombuffer(urandom(4*structure_size), dtype=np.uint32)
        p0r = np.frombuffer(urandom(4*structure_size), dtype=np.uint32)
        p1l = np.frombuffer(urandom(4*structure_size), dtype=np.uint32)
        p1r = np.frombuffer(urandom(4*structure_size), dtype=np.uint32)
        ks = gen_user_key(dis_nr + 2)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)
        tk0 = ks[-2][0]; tk1 = ks[-1][0]
        embedded_tk = combine_ints([extract_selected_bits_int(tk0, total_kg_bits_per_round[0], True), extract_selected_bits_int(tk1, total_kg_bits_per_round[1], True)], [len(total_kg_bits_per_round[0]), len(total_kg_bits_per_round[1])])
        start_time = time()
        kg_scores = key_guess_two_rounds((c0l, c0r, c1l, c1r), lookup_table, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, total_kg_bits_per_round, borrow_pos_per_round, m_per_round)
        end_time = time()
        wk_pos = np.ones(2**(len(total_kg_bits_0) + len(total_kg_bits_1)), dtype=bool)
        wk_pos[embedded_tk] = False
        wk_scores = kg_scores[wk_pos]
        wk_max_score = max(wk_max_score, np.max(wk_scores))
        tk_score = kg_scores[embedded_tk]
        tk_scores.append(tk_score)
        tk_ranks.append(np.sum(kg_scores > tk_score))
        wk_average_scores.append(np.mean(wk_scores))
        print("True key score: {}, true key rank: {}".format(tk_scores[-1], tk_ranks[-1]))
        print("Running time: {} s".format(end_time - start_time))
        sv_attack_time_consumption[try_index] = end_time - start_time
        surviving_kg_num = np.sum(kg_scores > c)
        if surviving_kg_num > 0:
            sv_whether_surviving_kg[try_index] = 1
            sv_num_surviving_kgs[try_index] = surviving_kg_num
            print("surviving key guess num:", surviving_kg_num)
        else:
            print("No surviving key guess")
        print('', flush=True)
    print("Average true key score:", np.mean(tk_scores))
    print("Average wrong key scores:", np.mean(wk_average_scores))
    print("Max wrong key scores:", wk_max_score)
    print("Average true key rank:", np.mean(tk_ranks))
    print("Median true key rank:", np.median(tk_ranks))
    print("Key surviving time:", np.sum(sv_whether_surviving_kg))
    print("Average attack time: {} s".format(np.mean(sv_attack_time_consumption)))
    print("Average surviving key guess number:", np.mean(sv_num_surviving_kgs))
    if not path.isdir(attack_saved_folder):
        makedirs(attack_saved_folder)
    np.save(attack_saved_folder + "tk_scores.npy", np.array(tk_scores))
    np.save(attack_saved_folder + "tk_rank.npy", np.array(tk_ranks))
    np.save(attack_saved_folder + 'sv_whether_surviving_kg.npy', sv_whether_surviving_kg)
    np.save(attack_saved_folder + 'sv_attack_time.npy', sv_attack_time_consumption)
    np.save(attack_saved_folder + 'sv_num_surviving_kgs.npy', sv_num_surviving_kgs)

def test_attack_with_NBs(n, NBs, pre_nr, pre_diff, lookup_table_path, selected_bits, max_num_plaintext_structure, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, borrow_pos_per_round, m_per_round, c):
    lookup_table = np.load(lookup_table_path)
    attack_nr = 1 + pre_nr + dis_nr + 2
    total_kg_bits_0 = [i for i in reversed(range(sp.WORD_SIZE())) if i in direct_kg_bits_per_round[0] or i in borrow_kg_bits_per_round_stage2[0]]
    total_kg_bits_1 = [i for i in reversed(range(sp.WORD_SIZE())) if i in direct_kg_bits_per_round[1] or i in borrow_kg_bits_stage1 or i in borrow_kg_bits_per_round_stage2[1]]
    total_kg_bits_per_round = [total_kg_bits_0, total_kg_bits_1]
    print("Total kg bits per round:", total_kg_bits_per_round)
    MAX_SUCCEED_KG_DIFF_HW = 2
    sv_whether_succeed = np.zeros(n, dtype=np.uint8)
    sv_whether_surviving_kg = np.zeros(n, dtype=np.uint8)
    sv_attack_time_consumption = np.zeros(n)
    sv_best_kg_diff_hw = []
    sv_num_used_structures = np.zeros(n, dtype=np.uint32)
    sv_best_kg_diff_show_time = np.zeros(1+len(total_kg_bits_0)+len(total_kg_bits_1), dtype=np.uint32)
    sv_num_surviving_kgs = np.zeros(n, dtype=np.uint32)
    for attack_index in range(n):
        print('Attack index:', attack_index)
        ks = gen_user_key(attack_nr)
        tk0 = ks[-2][0]; tk1 = ks[-1][0]
        embedded_tk = combine_ints([extract_selected_bits_int(tk0, total_kg_bits_per_round[0], True), extract_selected_bits_int(tk1, total_kg_bits_per_round[1], True)], [len(total_kg_bits_per_round[0]), len(total_kg_bits_per_round[1])])
        num_used_structures = 0
        start_time = time()
        while num_used_structures < max_num_plaintext_structure:
            num_used_structures += 1
            p0l, p0r, p1l, p1r = make_plaintext_structure(pre_diff, NBs)
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
            # Execute core attack process
            surviving_kgs_scores_list = guess_two_rounds_and_filter((c0l, c0r, c1l, c1r), lookup_table, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, total_kg_bits_per_round, borrow_pos_per_round, m_per_round, c)
            if surviving_kgs_scores_list:
                break
        end_time = time()
        sv_attack_time_consumption[attack_index] = end_time - start_time
        sv_num_used_structures[attack_index] = num_used_structures
        print("Running time: {} s".format(sv_attack_time_consumption[attack_index]))
        print("Used structure number:", num_used_structures)
        if surviving_kgs_scores_list:
            print("surviving key num:", len(surviving_kgs_scores_list))
            sv_num_surviving_kgs[attack_index] = len(surviving_kgs_scores_list)
            sv_whether_surviving_kg[attack_index] = 1
            best_embedded_kg = surviving_kgs_scores_list[0][0]
            hw = cal_hw(embedded_tk ^ best_embedded_kg, 32)
            sv_best_kg_diff_hw.append(hw)
            sv_best_kg_diff_show_time[hw] += 1
            print("Hamming distance between best key guess and true key:", hw)
            if hw <= MAX_SUCCEED_KG_DIFF_HW:
                sv_whether_succeed[attack_index] = 1
        else:
            print('No surviving key')
    print("Max succeed kg diff hw matric:", MAX_SUCCEED_KG_DIFF_HW)
    print("Key surviving time:", np.sum(sv_whether_surviving_kg))
    print("Attack success time:", np.sum(sv_whether_succeed))
    print("Average attack time: {} s".format(np.mean(sv_attack_time_consumption)))
    print("Average structure consumption:", np.mean(sv_num_used_structures))
    print("Average diff hw between true key and surviving key guess:", np.mean(sv_best_kg_diff_hw))
    print("Best kg diff hw show time:", sv_best_kg_diff_show_time)

if __name__ == "__main__":
    structure_size = 2**10
    dis_nr = 6
    diff_pos = 42
    selected_bits = [21, 20]
    direct_kg_bits_per_round = [[13,12], [16,15,8]]
    borrow_kg_bits_stage1 = [14,13,12]
    borrow_kg_bits_per_round_stage2 = [[11,10,9],[7,6,5,4,3,2,1,0]]
    borrow_pos_per_round = [12, 15]
    m_per_round = [3, 3]
    diff_l = 0
    diff_r = 0
    if diff_pos < sp.WORD_SIZE():
        diff_r |= 1 << diff_pos
    else:
        diff_l |= 1 << (diff_pos - sp.WORD_SIZE())
    dis_diff = (diff_l, diff_r)
    c = 10
    attack_saved_folder = 'attack_res/guess_two_rounds/'
    lookup_table_path = 'lookup_table/6/21-20_{}_student_distinguisher.npy'.format(diff_pos)
    # Test the key recovery attack using right plaintext structures
    test_attack(500, structure_size, dis_nr, dis_diff, lookup_table_path, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, borrow_pos_per_round, m_per_round, c, attack_saved_folder)
    # Test the key recovery attack using wrong plaintext structures
    test_attack_with_bad_structure(500, structure_size, dis_nr, dis_diff, lookup_table_path, selected_bits, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, borrow_pos_per_round, m_per_round, c, attack_saved_folder+'bad_structure/')

    # 12-round attack
    pre_nr = 3
    pre_diff = (0x92000204, 0x821002)
    dis_nr = 6
    NBs = [[17,57], [23,63],[27],[28],[29],[34],[35],[36],[37],[38]]
    selected_bits = [21, 20]
    direct_kg_bits_per_round = [[12], [15,7]]
    borrow_kg_bits_stage1 = [14,13,12]
    borrow_kg_bits_per_round_stage2 = [[11,10,9],[6,5,4,3,2,1,0]]
    borrow_pos_per_round = [12, 15]
    m_per_round = [3, 3]
    c = 10
    diff_pos = 42
    lookup_table_path = 'lookup_table/6/21-20_{}_student_distinguisher.npy'.format(diff_pos)
    max_num_plaintext_structure = 4308
    test_attack_with_NBs(10, NBs, pre_nr, pre_diff, lookup_table_path, selected_bits, max_num_plaintext_structure, direct_kg_bits_per_round, borrow_kg_bits_stage1, borrow_kg_bits_per_round_stage2, borrow_pos_per_round, m_per_round, c)