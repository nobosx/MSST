import student_net as sn
import speck as sp

# Train a 9-round neural network distinguisher
selected_bits = [21,20]
nr = 9
diff_pos = 66
diff_l, diff_r = 0, 0
if diff_pos < sp.WORD_SIZE():
    diff_r = 1 << diff_pos
else:
    diff_l = 1 << (diff_pos - sp.WORD_SIZE())
diff = (diff_l, diff_r)
saved_folder = './saved_model/9/21-20_{}_'.format(diff_pos)
sn.train_speck_distinguisher(num_epochs=10, num_rounds=nr, depth=1, diff=diff, bits=selected_bits, folder=saved_folder)