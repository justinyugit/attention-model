import pickle
f = open("data sample 3000.pkl", "rb")
l = pickle.load(f)
print(type(l['loc'][2]['loc']))