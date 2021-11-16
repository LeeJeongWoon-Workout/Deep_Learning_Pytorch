full_loss_list=[loss_list1,loss_list2,loss_list3,loss_list4,loss_list5,loss_list6,loss_list7,loss_list8]

for i in full_loss_list:
  plt.plot(i)


plt.legend(['elu','relu' ,'sigmoid', 'mish','hardtanh', 'tanh', 'selu','gelu'])
plt.xticks(rotation=90)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.figure(figsize=(20,10))
plt.show()

full_acc_list=[accuracy_list1,accuracy_list2,accuracy_list3,accuracy_list4,accuracy_list5,accuracy_list6,accuracy_list7,accuracy_list8]

for i in full_acc_list:
  plt.plot(i)

plt.legend(['elu','relu' ,'sigmoid', 'mish','hardtanh', 'tanh', 'selu','gelu'])
plt.xticks(rotation=90)
plt.xlabel("epoch")
plt.ylabel("Validation-Accuracy")
plt.figure(figsize=(20,10))
plt.show()
