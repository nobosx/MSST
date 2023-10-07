import speck as sp
import student_net as sn

# Train a 7-round neural network distinguisher
selected_bits = [21,20]
diff_pos = 42
nr = 7
saved_folder = './saved_model/7/21-20_{}_'.format(diff_pos)
diff_l, diff_r = 0, 0
if diff_pos < sp.WORD_SIZE():
    diff_r = 1 << diff_pos
else:
    diff_l = 1 << (diff_pos - sp.WORD_SIZE())
diff = (diff_l, diff_r)
sn.train_speck_distinguisher(10, nr, 1, diff, selected_bits, saved_folder)