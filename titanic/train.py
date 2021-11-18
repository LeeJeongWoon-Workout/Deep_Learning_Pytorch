import tqdm

EPOCHS  = 1000

loss_list     = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(train_data.float())
    loss = loss_fn(y_pred, train_label.float())
    loss_list[epoch] = loss
    
    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        y_pred = model(test_data.float())
        correct = (torch.argmax(y_pred, dim=1) == test_label.float()).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()
