import sys
sys.stdout = open('file_out.txt', 'w')
print('Hello World!')
sys.stdout.close()

# test_file_redirect.txt', 'w')