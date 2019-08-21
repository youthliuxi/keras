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

def drawPlt_batch_size(optimizer, init_mode, batch_sizes, data):
	data_str = ['val_loss','val_acc','loss','acc']
	color = ['red','orange','yellow','green','cyan','blue','purple','black']
	plt.figure(figsize=(10,8))
	for i in range(0,4):
		plt.subplot(221+i)
		# plt.plot(data[data_str[i]])
		for x in range(0,8):
			plt.plot(data[batch_sizes[x]][data_str[i]],color=color[x],linewidth=1)
		plt.legend(batch_sizes, loc = 'center right')
		plt.title(data_str[i])
	# plt.show()
	jpgName = '%s-%s' % (optimizer, init_mode)
	plt.savefig("picture/batch_size/%s.png" % (jpgName))
	print("picture/batch_size/%s.png" % (jpgName))
	plt.close()

def drawPlt_optimizer(batch_size, init_mode, optimizers, data):
	data_str = ['val_loss','val_acc','loss','acc']
	color = ['red','orange','yellow','green','cyan','blue','purple','black']
	plt.figure(figsize=(10,8))
	for i in range(0,4):
		plt.subplot(221+i)
		# plt.plot(data[data_str[i]])
		for x in range(0,7):
			plt.plot(data[optimizers[x]][data_str[i]],color=color[x],linewidth=1)
		plt.legend(optimizers, loc = 'center right')
		plt.title(data_str[i])
	# plt.show()
	jpgName = '%s-%s' % (init_mode, batch_size)
	plt.savefig("picture/optimizer/%s.png" % (jpgName))
	print("picture/optimizer/%s.png" % (jpgName))
	plt.close()

def drawPlt_init_mode(optimizer, batch_size, init_modes, data):
	data_str = ['val_loss','val_acc','loss','acc']
	color = ['red','orange','yellow','green','cyan','blue','purple','black']
	plt.figure(figsize=(10,8))
	for i in range(0,4):
		plt.subplot(221+i)
		# plt.plot(data[data_str[i]])
		for x in range(0,7):
			plt.plot(data[init_modes[x]][data_str[i]],color=color[x],linewidth=1)
		plt.legend(init_modes, loc = 'center right')
		plt.title(data_str[i])
	# plt.show()
	jpgName = '%s-%s' % (batch_size, optimizer)
	plt.savefig("picture/init_mode/%s.png" % (jpgName))
	print("picture/init_mode/%s.png" % (jpgName))
	plt.close()

def main():
	batch_sizes = [16,32,64,128,256,512,1024,2048,4096]
	init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	optimizers = ["SGD","RMSprop","Adagrad","Adadelta","Adam","Adamax","Nadam"]

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
			drawPlt_batch_size(optimizer, init_mode, batch_sizes, data)
	for batch_size in batch_sizes:
		for init_mode in init_modes:
			data = {}
			for optimizer in optimizers:
				with open("txt/%s-%s-%s.txt" % (init_mode,optimizer,batch_size), 'r', encoding = 'utf-8') as f:
					file_txt = f.read()
					file_txt = re.sub('\'','\"',file_txt)
					file_txt_data = json.loads(file_txt)
					data[optimizer] = file_txt_data
			drawPlt_optimizer(batch_size, init_mode, optimizers, data)

	for optimizer in optimizers:
		for batch_size in batch_sizes:
			data = {}
			for init_mode in init_modes:
				with open("txt/%s-%s-%s.txt" % (init_mode,optimizer,batch_size), 'r', encoding = 'utf-8') as f:
					file_txt = f.read()
					file_txt = re.sub('\'','\"',file_txt)
					file_txt_data = json.loads(file_txt)
					data[init_mode] = file_txt_data
			drawPlt_init_mode(optimizer, batch_size, init_modes, data)

if __name__ == '__main__':
	main()
