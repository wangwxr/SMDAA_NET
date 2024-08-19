import matplotlib.pyplot as plt
class visual:
    def __init__(self,path,totalepoch):
        self.savepath = path
        self.totalepoch = totalepoch
        self.train_dice_lines = []
        self.val_dice_lines =  []
        # self.train_acc_list=[]
        # self.val_acc_list=[]
        self.train_dice_list=[]
        self.val_dice_list=[]
        self.x = []
    def visual(self,epoch,train_dice,val_dice):

        self.x.append(epoch)
        self.train_dice_list.append(train_dice)
        self.val_dice_list.append(val_dice)
        # self.train_acc_list.append(train_se)
        # self.val_acc_list.append(val_se)
        if epoch == self.totalepoch - 1:
            plt.figure(figsize=(8, 6), dpi=100)
            # 创建两行一列的图，并指定当前使用第一个图
            plt.subplot(1, 1, 1)
            # try:
            #     self.train_dice_lines.remove(self.train_dice_lines[0])  # 移除上一步曲线
            #     self.val_dice_lines.remove(self.val_dice_lines[0])
            # except Exception:
            #     pass

            self.train_dice_lines = plt.plot(self.x, self.train_dice_list, 'r', lw=1)  # lw为曲线宽度
            self.val_dice_lines = plt.plot(self.x, self.val_dice_list, 'b', lw=1)
            plt.title("dice")
            plt.xlabel("epoch")
            plt.ylabel("dice")
            plt.legend(["train_dice",
                        "val_dice"])

            plt.savefig(f'{self.savepath}/savefig_example_rotate.png')

        # # 创建两行一列的图，并指定当前使用第二个图
        # plt.subplot(2, 1, 2)
        # try:
        #     self.train_acc_lines.remove(self.train_acc_lines[0])  # 移除上一步曲线
        #     self.val_acc_lines.remove(self.val_acc_lines[0])
        # except Exception:
        #     pass
        #
        # train_acc_lines = plt.plot(self.x, self.train_acc_list, 'r', lw=1)  # lw为曲线宽度
        # val_acc_lines = plt.plot(self.x, self.val_acc_list, 'b', lw=1)
        # plt.title("acc")
        # plt.xlabel("epoch")
        # plt.ylabel("acc")
        # plt.legend(["train_acc",
        #             "val_acc"])

