# -*- coding:utf-8 -*-
import os
import matplotlib
import json
import re
import matplotlib.pyplot as plt
# def listdir(path, list_name):  #传入存储的list
# 	for file in os.listdir(path):  
# 		file_path = os.path.join(path, file)  
# 		if os.path.isdir(file_path):  
# 			listdir(file_path, list_name)  
# 		else:  
# 			list_name.append(file_path)

# 	return list_name

def drawPlt(optimizer, init_mode, batch_sizes, data):
	data_str = ['val_loss','val_acc','loss','acc']
	color = ['red','orange','yellow','green','cyan','blue','purple','black']
	plt.figure(figsize=(10,8))
	for i in range(0,4):
		plt.subplot(221+i)
		# plt.plot(data[data_str[i]])
		for x in range(0,8):
			plt.plot(data[batch_sizes[x]][data_str[i]],color=color[x],linewidth=1)
		plt.title(data_str[i])
	# plt.show()
	jpgName = '%s-%s' % (optimizer, init_mode)
	plt.savefig("picture\\%s.png" % (jpgName))
	print("picture\\%s.png" % (jpgName))
	plt.close()

def main():
	batch_sizes = [16,32,64,128,256,512,1024,2048]
	init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	optimizers = ["SGD","RMSprop","Adagrad","Adadelta","Adam","Adamax","Nadam"]
	path_index = "txt"
	path = "picture"
	text = os.listdir(path)
	for optimizer in optimizers:
		for init_mode in init_modes:
			data = {}
			for batch_size in batch_sizes:
				with open("txt/%s-%s-%s.txt" % (init_mode,optimizer,batch_size), 'r', encoding = 'utf-8') as f:
					file_txt = f.read()
					file_txt = re.sub('\'','\"',file_txt)
					file_txt_data = json.loads(file_txt)
					data[batch_size] = file_txt_data
			
			# print(data[1024]['acc'][-1])
			# print(data.keys())
			drawPlt(optimizer, init_mode, batch_sizes, data)

if __name__ == '__main__':
	main()