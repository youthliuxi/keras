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

def updateFileName(text_name):
	# 更新文件名中的参数排列顺序和分隔符
	file_name = text_name.split(".")[0]
	name_part = file_name.split("_")
	optimizer = name_part[0]
	batch_size = name_part[-1]
	init_mode = file_name.lstrip("%s" % optimizer).rstrip("%s" % batch_size).strip("_")
	print(init_mode)
	os.system("mv %s\\%s_%s_%s.txt %s\\%s-%s-%s.txt" % (path, optimizer, init_mode, batch_size, path, init_mode, optimizer, batch_size))
	print("mv %s\\%s_%s_%s.txt %s\\%s-%s-%s.txt" % (path, optimizer, init_mode, batch_size, path, init_mode, optimizer, batch_size))
	# 更新文件名结束
def splitFileName(text_name):
	file_name = text_name.split(".")[0]
	name_part = file_name.split("-")
	init_mode = name_part[0]
	optimizer = name_part[1]
	batch_size = name_part[2]
	return init_mode, optimizer, batch_size

def drawPlt(path_index,data,jpgName):
	data_str = ['val_loss','val_acc','loss','acc']
	plt.figure(figsize=(10,8))
	for i in range(0,4):
		plt.subplot(221+i)
		plt.plot(data[data_str[i]])
		plt.title(data_str[i])
	# plt.show()
	plt.savefig("%s\\picture\\%s.png" % (path_index,jpgName))
	print("%s\\picture\\%s.png" % (path_index,jpgName))
	plt.close()

def main():
	path_index = "try_third"
	path = "try_third\\txt"
	text = os.listdir(path)
	for text_name in text:
		# print(text_name)
		# updateFileName(text_name)
		init_mode, optimizer, batch_size = splitFileName(text_name)
		# print(init_mode)
		with open("%s\\%s" % (path, text_name), 'r', encoding = 'utf-8') as f:
			# print(f)
			file_txt = f.read()
			file_txt = re.sub('\'','\"',file_txt)
			# 将单引号替换双引号
			data =json.loads(file_txt)
			# print(data['val_loss'])
			jpgName = "%s-%s-%s" % (init_mode,optimizer,batch_size)
			drawPlt(path_index,data,jpgName)
		# break
		

if __name__ == '__main__':
	main()