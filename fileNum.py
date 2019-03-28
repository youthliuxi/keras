# coding=utf-8
import os
import sys
import glob

batch_sizes = [16,32,64,128]
init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizers = ["SGD","RMSprop","Adagrad","Adadelta","Adam","Adamax","Nadam"]
i=0
for optimizer in optimizers:
	for init_mode in init_modes:
		for batch_size in batch_sizes:
			i = i + 1

print(i)

# path_file_number=glob.glob('D:/case/test/testcase/checkdata/*.py')#或者指定文件下个数
path_file_number=glob.glob(pathname='try_second/*.py') #获取当前文件夹下个数
# print(path_file_number)
print(len(path_file_number))