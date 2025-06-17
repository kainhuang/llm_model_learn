import util
import sys
from util import dic2json
import json

for line in sys.stdin:
    dic = json.loads(line.strip())
    print (dic['res'])