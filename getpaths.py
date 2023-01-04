import os

n = 1

toppath = r'./data/raw_data_subset_500_samples/'
list_of_files = []

f = open("paths.txt", "w")
for i in range(n):
	path = toppath + str(500 * i) + "/"
	for root, dirs, files in os.walk(path):
		for file in files:
			list_of_files.append(os.path.join(root,file))
for name in list_of_files:
    f.write(name + "\n")
