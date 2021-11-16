import tqdm

EPOCHS  = 40
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

X_train=X_train.cuda()
y_train=y_train.cuda()
X_test=X_test.cuda()
y_test=y_test.cuda()

loss_list1     = np.zeros((EPOCHS,))
accuracy_list1 = np.zeros((EPOCHS,))

loss_list2     = np.zeros((EPOCHS,))
accuracy_list2 = np.zeros((EPOCHS,))

loss_list3     = np.zeros((EPOCHS,))
accuracy_list3 = np.zeros((EPOCHS,))

loss_list4     = np.zeros((EPOCHS,))
accuracy_list4 = np.zeros((EPOCHS,))

loss_list5     = np.zeros((EPOCHS,))
accuracy_list5 = np.zeros((EPOCHS,))

loss_list6     = np.zeros((EPOCHS,))
accuracy_list6 = np.zeros((EPOCHS,))

loss_list7     = np.zeros((EPOCHS,))
accuracy_list7 = np.zeros((EPOCHS,))

loss_list8     = np.zeros((EPOCHS,))
accuracy_list8 = np.zeros((EPOCHS,))


for epoch in tqdm.trange(EPOCHS):
    y_pred1 = model1(X_train)
    loss1 = loss_fn1(y_pred1, y_train)
    loss_list1[epoch] = loss1.item()
    
    # Zero gradients
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    
    with torch.no_grad():
        y_pred1 = model1(X_test)
        correct = (torch.argmax(y_pred1, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list1[epoch] = correct.mean()


    y_pred2 = model2(X_train)
    loss2 = loss_fn2(y_pred2, y_train)
    loss_list2[epoch] = loss2.item()
    
    # Zero gradients
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    
    with torch.no_grad():
        y_pred2 = model2(X_test)
        correct = (torch.argmax(y_pred2, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list2[epoch] = correct.mean()


    y_pred3 = model3(X_train)
    loss3 = loss_fn3(y_pred3, y_train)
    loss_list3[epoch] = loss3.item()
    
    # Zero gradients
    optimizer3.zero_grad()
    loss3.backward()
    optimizer3.step()
    
    with torch.no_grad():
        y_pred3 = model3(X_test)
        correct = (torch.argmax(y_pred3, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list3[epoch] = correct.mean()

    y_pred4 = model4(X_train)
    loss4 = loss_fn4(y_pred4, y_train)
    loss_list4[epoch] = loss4.item()
    
    # Zero gradients
    optimizer4.zero_grad()
    loss4.backward()
    optimizer4.step()
    
    with torch.no_grad():
        y_pred4 = model4(X_test)
        correct = (torch.argmax(y_pred4, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list4[epoch] = correct.mean()

    y_pred5 = model5(X_train)
    loss5 = loss_fn5(y_pred5, y_train)
    loss_list5[epoch] = loss5.item()
    
    # Zero gradients
    optimizer5.zero_grad()
    loss5.backward()
    optimizer5.step()
    
    with torch.no_grad():
        y_pred5 = model5(X_test)
        correct = (torch.argmax(y_pred5, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list5[epoch] = correct.mean()

    y_pred6 = model6(X_train)
    loss6 = loss_fn6(y_pred6, y_train)
    loss_list6[epoch] = loss6.item()
    
    # Zero gradients
    optimizer6.zero_grad()
    loss6.backward()
    optimizer6.step()
    
    with torch.no_grad():
        y_pred6 = model6(X_test)
        correct = (torch.argmax(y_pred6, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list6[epoch] = correct.mean()

    y_pred7 = model7(X_train)
    loss7 = loss_fn7(y_pred7, y_train)
    loss_list7[epoch] = loss7.item()
    
    # Zero gradients
    optimizer7.zero_grad()
    loss7.backward()
    optimizer7.step()
    
    with torch.no_grad():
        y_pred7 = model7(X_test)
        correct = (torch.argmax(y_pred7, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list7[epoch] = correct.mean()

    y_pred8 = model8(X_train)
    loss8 = loss_fn8(y_pred8, y_train)
    loss_list8[epoch] = loss8.item()

    optimizer8.zero_grad()
    loss8.backward()
    optimizer8.step()
    
    with torch.no_grad():
        y_pred8 = model8(X_test)
        correct = (torch.argmax(y_pred8, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list8[epoch] = correct.mean()
