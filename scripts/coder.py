import pickle
#a = [4,5,12,16,20,23,29,37,42,48,52,58,63,67,67,71,71,73,73,74,76,79,79,82,82,86,91,92,94,96,97,98,99,99,99,99,100,100,100,100,100,100,100]
#b = [0,0,0,0,0,0,0,0,1,1,1,2,3,5,8,9,12,15,18,21,23,24,24,24,24,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25]
#c = [3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,3,4,7,7,8,9,9,10,11,13,14,15,17,18,19,19,20,23,26,29,31,33,36,37,40,41,43,46,47,48,49]
d = [4,5,6,7,6,7,9,7,9,9,8,9,9,9,10,10,10,11,11,9,12,12,12,12,12,13,14,14,13,13,13,12]
fobj = open("complex4state_metaCorNoise.csv", "wb")
pickle.dump(d, fobj)
fobj.close()