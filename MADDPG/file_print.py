import sys
print('prima di cambio')

sys.stdout = open('file_out1.txt', 'w')
print('mio HW')
