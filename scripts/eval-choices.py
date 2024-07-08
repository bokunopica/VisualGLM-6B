import random
import pickle

def main():
    a = []
    for i in range(10):
        a.append([])
        for j in range(146):
            a[i].append(j)
        random.shuffle(a[i])

    for i in range(10):
        for j in range(i+1, 10):
            if str(a[i]) == str(a[j]):
                print("duplicate")
                return 0
    with open("bootstrap_index.list", 'wb') as f:
        pickle.dump(a, f)
            
main()

