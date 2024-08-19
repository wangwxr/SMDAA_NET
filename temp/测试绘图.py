from utils.visualization import *
vis = visual('./', 10)
for i in range(10):
     vis.visual(epoch=i,train_dice=i*0.1,val_dice=i*0.08)