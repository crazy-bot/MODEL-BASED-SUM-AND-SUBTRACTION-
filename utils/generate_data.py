
from collections import defaultdict
import numpy as np
import pandas as pd

def comb_sum(target, s, pairs):
    if s == target and len(pairs)==2:
        map_sum[s].append(pairs)
        #print(s, pairs)
        return
    if s > target or len(pairs)==2:
        return

    for i in range(10):
        comb_sum(target, s+i, pairs+[i])

def comb_sub(target, s, pairs):
    if s == target and len(pairs)==2:
        map_sub[s].append(pairs)
        #print(s, pairs)
        return
    if s < target or len(pairs)==2:
        return

    for i in range(10):
        comb_sub(target, s-i, pairs+[i])

def calculate_weights(labels):
    counts = np.array([len(v) for k,v in labels.items()])
    class_weights = (sum(counts)/counts)/sum(counts)
    class_weights = class_weights.astype(np.float32)
    return class_weights

if __name__ == '__main__':
    map_sum = defaultdict(list)
    map_sub = defaultdict(list)
    for t in range(19):
        comb_sum(t, 0, [])
    #print(map_sum)

    for t in range(-9,10):
        for i in range(10):
            comb_sub(t, i, [i])
    #print(map_sub)

    sum_weights = calculate_weights(map_sum)

    outfile = open( 'data/map_sum.txt', 'w' )
    for key, value in sorted( map_sum.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )
    outfile.close()
    np.save('data/map_sum.npy', sum_weights)

    w1 = np.load('data/map_sum.npy', allow_pickle=True)

    sub_weights = calculate_weights(map_sum)

    outfile = open( 'data/map_sub.txt', 'w' )
    for key, value in sorted( map_sub.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )
    outfile.close()
    np.save('data/map_sub.npy', sub_weights)

    w2 = np.load('data/map_sub.npy', allow_pickle=True)

    print(w1, w2 )

    
    
   





