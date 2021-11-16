#elu relu sigmoid mish tanh selu tanh hardtanh step

model1     = Model1(X_train.shape[1])
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
loss_fn1   = nn.CrossEntropyLoss()

model2     = Model2(X_train.shape[1])
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
loss_fn2   = nn.CrossEntropyLoss()

model3     = Model3(X_train.shape[1])
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
loss_fn3   = nn.CrossEntropyLoss()

model4     = Model4(X_train.shape[1])
optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.001)
loss_fn4   = nn.CrossEntropyLoss()

model5     = Model5(X_train.shape[1])
optimizer5 = torch.optim.Adam(model5.parameters(), lr=0.001)
loss_fn5   = nn.CrossEntropyLoss()

model6     = Model6(X_train.shape[1])
optimizer6 = torch.optim.Adam(model6.parameters(), lr=0.001)
loss_fn6   = nn.CrossEntropyLoss()

model7     = Model7(X_train.shape[1])
optimizer7 = torch.optim.Adam(model7.parameters(), lr=0.001)
loss_fn7   = nn.CrossEntropyLoss()

model8     = Model8(X_train.shape[1])
optimizer8 = torch.optim.Adam(model8.parameters(), lr=0.001)
loss_fn8   = nn.CrossEntropyLoss()

model1=model1.cuda()
model2=model2.cuda()
model3=model3.cuda()
model4=model4.cuda()
model5=model5.cuda()
model6=model6.cuda()
model7=model7.cuda()
model8=model8.cuda()
