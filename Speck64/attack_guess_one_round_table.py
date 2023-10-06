import numpy as np
import speck as sp
import math
from time import time
from os import urandom, path, makedirs
from util import cal_hw, extract_bits_and_combine

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

def generate_user_key(nr, master_key_bit_length=96):
    WORD_SIZE = sp.WORD_SIZE()
    assert master_key_bit_length % WORD_SIZE == 0
    m = master_key_bit_length // WORD_SIZE
    assert m == 3 or m == 4
    key = np.frombuffer(urandom(m * 4), dtype=np.uint32).reshape(m, 1)
    ks = sp.expand_key(key, nr)
    return ks

def expand_key_guess_mask(key_guess_bits):
    kg = np.zeros((1,), dtype=np.uint32)
    for kg_bit in key_guess_bits:
        mask = 1 << kg_bit
        kg = np.concatenate((kg, kg ^ mask))
    assert kg[0] == 0
    return kg

# Try to determine the borrow bit b[kg_low] at bit position kg_low using m consecutive bits [kg_low-1~kg_low-m]
# Return: b[kg_low], is_valid
# is_valid: whether the borrow bit at bit position kg_low of a ciphertext can be recovered
def determine_borrow_bit_from_low(z, y, kg_low, m):
    mask_val = (1 << m) - 1
    a = (z >> (kg_low - m)) & mask_val
    b = (y >> (kg_low - m)) & mask_val
    return a < b, a != b

# Core attack process
# The filtering process ueses a lookup table as a distingusiher
# The total key guessing space is divided into two stages
# Stage1: guess the subkey bits that are related to the borrow bit in the modular substratcion operation of the last round
# Stage2: guess other subkey bits that are related to the input of the distinguisher
def guess_last_round_sk_and_filter(ciphertext_structure, lookup_table, selected_bits, kg_low, m, key_guess_bits, c):
    c0l, c0r, c1l, c1r = ciphertext_structure
    structure_size = len(c0l)
    direct_key_guess_space = 2**len(key_guess_bits)
    direct_kg = expand_key_guess_mask(key_guess_bits)
    c0r = c0l ^ c0r; c1r = c1l ^ c1r
    y0 = sp.ror(c0r, sp.BETA()); y1 = sp.ror(c1r, sp.BETA())
    surviving_kgs_scores_list = []
    # Stage1
    for i in range(2**m):
        additional_kg = i << (kg_low - m)
        partial_z0 = c0l ^ additional_kg; partial_z1 = c1l ^ additional_kg
        _, is_valid0 = determine_borrow_bit_from_low(partial_z0, y0, kg_low, m)
        _, is_valid1 = determine_borrow_bit_from_low(partial_z1, y1, kg_low, m)
        valid_pair_pos = is_valid0 & is_valid1
        valid_num = np.sum(valid_pair_pos)
        valid_pr = valid_num / structure_size
        # Find ciphertext pairs whose borrow bit can be recovered
        partial_z0 = partial_z0[valid_pair_pos]; partial_z1 = partial_z1[valid_pair_pos]
        valid_y0 = y0[valid_pair_pos]; valid_y1 = y1[valid_pair_pos]
        # Stage2
        repeated_direct_kg = np.repeat(direct_kg, valid_num)
        partial_z0 = np.tile(partial_z0, direct_key_guess_space); partial_z1 = np.tile(partial_z1, direct_key_guess_space)
        valid_y0 = np.tile(valid_y0, direct_key_guess_space); valid_y1 = np.tile(valid_y1, direct_key_guess_space)
        partial_z0 = partial_z0 ^ repeated_direct_kg; partial_z1 = partial_z1 ^ repeated_direct_kg
        x0 = partial_z0 - valid_y0; x1 = partial_z1 - valid_y1
        x0 = sp.rol(x0, sp.ALPHA()); x1 = sp.rol(x1, sp.ALPHA())
        table_input_int = extract_bits_and_combine([x0, valid_y0, x1, valid_y1], selected_bits)
        Z = lookup_table[table_input_int].reshape(direct_key_guess_space, valid_num)
        Z = np.sum(Z, axis=1) / valid_pr
        surviving_direct_kgs = direct_kg[Z > c]
        if len(surviving_direct_kgs) > 0:
            surviving_direct_kgs_scores = Z[Z > c]
            for j in range(len(surviving_direct_kgs)):
                surviving_kgs_scores_list.append((surviving_direct_kgs[j] ^ additional_kg, surviving_direct_kgs_scores[j]))
    if len(surviving_kgs_scores_list) > 0:
        surviving_kgs_scores_list.sort(key=lambda x: x[1], reverse=True)
    return surviving_kgs_scores_list

# Execute key recovery attack of 1+pre_nr+dis_nr+1 rounds and recover certain subkey bits of the last round
# Generate plaintext structures using neutral bits
def key_recovery_attack_with_NBs(n, lookup_table_path, dis_nr, dis_diff, selected_bits, plain_diff, NBs, max_num_plaintext_structure, pre_nr, kg_low, m, key_guess_bits, c, attack_saved_folder='./'):
    lookup_table = np.load(lookup_table_path)
    attack_nr = 1 + pre_nr + dis_nr + 1
    structrue_size = 2**len(NBs)
    kg_pos_mask = 0
    for i in key_guess_bits:
        kg_pos_mask |= 1 << i
    for i in range(kg_low - m, kg_low):
        kg_pos_mask |= 1 << i
    MAX_SUCCEED_KG_DIFF_HW = 2
    sv_whether_succeed = np.zeros(n, dtype=np.uint8)
    sv_whether_surviving_kg = np.zeros(n, dtype=np.uint8)
    sv_attack_time_consumption = np.zeros(n)
    sv_best_kg_diff_hw = []
    sv_num_used_structures = np.zeros(n, dtype=np.uint32)
    sv_best_kg_diff_show_time = np.zeros(1+m+len(key_guess_bits), dtype=np.uint32)
    sv_num_surviving_kgs = np.zeros(n, dtype=np.uint32)
    for attack_index in range(n):
        print('Attack index:', attack_index)
        user_key = generate_user_key(attack_nr)
        tk = user_key[-1][0] & kg_pos_mask
        num_used_structures = 0
        start_time = time()
        while num_used_structures < max_num_plaintext_structure:
            # Generate one plaintext structure and collect the corresponding ciphertext structure
            num_used_structures += 1
            p0l, p0r, p1l, p1r = make_plaintext_structure(plain_diff, NBs)
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, user_key)
            # Execute core attack pipeline
            surviving_kgs_scores_list = guess_last_round_sk_and_filter((c0l, c0r, c1l, c1r), lookup_table, selected_bits, kg_low, m, key_guess_bits, c)
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
            best_kg = surviving_kgs_scores_list[0][0]
            hw = cal_hw(tk ^ best_kg, 32)
            sv_best_kg_diff_hw.append(hw)
            sv_best_kg_diff_show_time[hw] += 1
            print('Diff between true key and best key guess:', hex(tk ^ best_kg))
            if hw <= MAX_SUCCEED_KG_DIFF_HW:
                sv_whether_succeed[attack_index] = 1
        else:
            print('No surviving key')
        print('', flush=True)
    print("Max succeed kg diff hw matric:", MAX_SUCCEED_KG_DIFF_HW)
    print("Key surviving time:", np.sum(sv_whether_surviving_kg))
    print("Attack success time:", np.sum(sv_whether_succeed))
    print("Average attack time: {} s".format(np.mean(sv_attack_time_consumption)))
    print("Average structure consumption:", np.mean(sv_num_used_structures))
    print("Average diff hw between true key and surviving key guess:", np.mean(sv_best_kg_diff_hw))
    print("Best kg diff hw show time:", sv_best_kg_diff_show_time)
    if not path.isdir(attack_saved_folder):
        makedirs(attack_saved_folder)
    np.save(attack_saved_folder + 'sv_whether_surviving_kg.npy', sv_whether_surviving_kg)
    np.save(attack_saved_folder + 'sv_whether_succeed.npy', sv_whether_succeed)
    np.save(attack_saved_folder + 'sv_attack_time.npy', sv_attack_time_consumption)
    np.save(attack_saved_folder + 'sv_num_used_structures.npy', sv_num_used_structures)
    np.save(attack_saved_folder + 'sv_best_kg_diff_show_time.npy', sv_best_kg_diff_show_time)

if __name__ == '__main__':
    kg_low = 24
    m = 3
    dis_nr = 7
    dis_diff = (0x40000000, 0)
    plain_diff = (0x48, 0x8000000)
    pre_nr = 1
    NBs = list(range(49, 39, -1))   # Using 10 neutral bits
    lookup_table_path = 'lookup_table/7/12-11-4-3-1-0_62_student_distinguisher.h5.npy'
    selected_bits = [12,11,4,3,1,0]
    key_guess_bits = [28,27,26,25,24, 4,3,2,1,0]                   
    c = 10
    max_num_plaintext_structure = 8
    attack_saved_folder = './attack_res/guess_one_round/'
    key_recovery_attack_with_NBs(1000, lookup_table_path, dis_nr, dis_diff, selected_bits, plain_diff, NBs, max_num_plaintext_structure, pre_nr, kg_low, m, key_guess_bits, c, attack_saved_folder)