import time
time_start = time.time()
time_end=time.time()
f =  open("time.txt",'w')
f.write(str(time_start)+'\n')
f.write(str(time_end)+'\n')
f.write(str(time_end-time_start)+'\n')
