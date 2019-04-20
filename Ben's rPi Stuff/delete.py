import os

path = os.path.realpath(__file__)
print(path[:len(path) - path[::-1].index('/') - 1])
