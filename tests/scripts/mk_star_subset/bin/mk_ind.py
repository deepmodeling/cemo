import pickle


def write_pkl(f: str, data: object):
    with open(f, "wb") as OUT:
        pickle.dump(data, OUT)


def main():
    ind = list(range(1, 10, 2))
    print(ind)
    f = "data/ind1.pkl"
    write_pkl(f, ind)
    print(f)


main()
