import sys
import json
from util import dic2json

if __name__ == '__main__':
    input_file_name = sys.argv[1]
    for line in open(input_file_name):
        dic = json.loads(line.strip())
        print (dic2json(dic))