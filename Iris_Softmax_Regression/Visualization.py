fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list3)
ax1.set_ylabel("validation accuracy")
ax2.plot(loss_list3)
ax2.set_ylabel("validation loss")
ax2.set_xlabel("epochs");
