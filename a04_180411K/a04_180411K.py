import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# Defining function for data preprocessing
def preprocessing(normalize,reshape):
    # Importing data set CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    K = len(np.unique(y_train)) # no.of Classes 
    Ntr = x_train.shape[0] # Number of training data=50,000
    Nte = x_test.shape[0] # Number of testing data=10,000
    Din = 3072 # CIFAR10 = 3072 = 32x32x3

    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0# Normalize pixel values

    # Center the pixel values
    mean_image = np.mean(x_train, axis=0)
    x_train = x_train - mean_image
    x_test = x_test - mean_image
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=K)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=K)

    #flatterning the input images
    if reshape:
        x_train = np.reshape(x_train,(Ntr,Din))
        x_test = np.reshape(x_test,(Nte,Din))
    # Changing the data types
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train,y_train,x_test,y_test,K,Din,Ntr,Nte


# Defining linear Classifier function
def layer1LinearClassifier(x_train,y_train,x_test,y_test,K,Din,lr,lr_decay,reg,Ntr,Nte):
    batch_size = Ntr
    loss_history = []
    loss_history_testing = []
    train_acc_history = []
    val_acc_history = []
    lr_array=[]
    seed = 0
    rng = np.random.default_rng(seed=seed)

    # Initializing weight and bias arrays
    std=1e-5#standard deviation to generate random values for w1 and b1
    w1 = std*np.random.randn(Din, K)#weight matrix
    b1 = np.zeros(K)#k dimensional bias matrix

    for t in range(iterations):
        # To prevent over fitting we shuffle the training data set in order to randomize the training process.
        indices = np.arange(Ntr)
        rng.shuffle(indices)
        x=x_train[indices]
        y=y_train[indices]

        # forward propagation
        y_pred=x.dot(w1)+b1
        y_pred_test=x_test.dot(w1)+b1
        val=y_pred_test.shape[0]
        # calculating loss using regularized loss function
        train_loss=(1/batch_size)*(np.square(y_pred-y)).sum()+reg*(np.sum(w1*w1))
        loss_history.append(train_loss)
        test_loss=(1/val)*(np.square(y_pred_test-y_test)).sum()+reg*(np.sum(w1*w1))
        loss_history_testing.append(test_loss)
        
        # calculating trainning and testing accuracies
        train_accuracy=1-(1/(10*batch_size))*(np.abs(np.argmax(y,axis=1)-np.argmax(y_pred,axis=1))).sum()
        train_acc_history.append(train_accuracy)

        test_accuracy=1-(1/(10*Nte))*(np.abs(np.argmax(y_test,axis=1)-np.argmax(y_pred_test,axis=1))).sum()
        val_acc_history.append(test_accuracy)

        if t%10 == 0:
            print('epoch %d/%d: train loss= %f || ,test loss= %f ||,train accuracy= %f ||, test accuracy= %f ||, learning rate= %f ||' 
            % (t,iterations,train_loss,test_loss,train_accuracy,test_accuracy,lr))

        # Backward propagation
        dy_pred=(1./batch_size)*2.0*(y_pred-y)#partial deravative of L w.r.t y_pred
        dw1=x.T.dot(dy_pred)+reg*w1
        db1=dy_pred.sum(axis=0)

        
        w1-=lr*dw1#update weight matrix
        b1-=lr*db1#update bias matrix
        lr_array.append(lr)
        lr*=lr_decay#decaying the learning rate
    return w1,b1,loss_history,loss_history_testing,train_acc_history,val_acc_history,lr_array


#defining parameters 
iterations = 300#gradient descent iterations
lr = 1.4e-2#learning rate
lr_decay= 0.999
reg = 5e-6 #lambda=regularization parameter
x_train,y_train,x_test,y_test,K,Din,Ntr,Nte=preprocessing(normalize=True,reshape=True)
 #Run the linear classifier
w1,b1,loss_history,loss_history_test,train_acc_history,val_acc_history,lr_array=layer1LinearClassifier(x_train,y_train,x_test,y_test,K,Din,lr,lr_decay,reg,Ntr,Nte)

# ploting the graphs of training and testing losses, training and testing accuracies and learning rate
fig, axes  = plt.subplots(1,5)
titles = {"Training Loss":loss_history, "testing loss":loss_history_test,"Training Accuracy":train_acc_history,
         "testing Accuracy": val_acc_history, "Learning Rate":lr_array}
place = 0
for key in titles.keys():
    axes[place].plot(titles[key])
    axes[place].set_title(key)
    place+=1
plt.show()
#plotting the weight matrix W as 10 images
images=[]
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']#classes of cifar-10
for i in range(w1.shape[1]):
    reshapen=np.reshape(w1[:,i]*255,(32,32,3))
    normalized=cv.normalize(reshapen, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    images.append(normalized)
fig,ax=plt.subplots(2,5,figsize=(30,10))
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(images[i*5+j],vmin=0,vmax=255)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_title(classes[i*5+j])
plt.show()


# Function for two layer dense network
def layer_2(x_train,y_train,x_test,y_test,Din,lr,lr_decay,H,reg,K,Ntr,Nte):
    loss_history = []
    loss_history_test = []
    train_acc_history = []
    val_acc_history = []
    lr_array =[]
    seed = 0
    rng = np.random.default_rng(seed=seed)
    batch_size=Ntr

    std=1e-5
    #initializing weight and bias matrices for hidden layer
    w1 = std*np.random.randn(Din, H)
    b1 = np.zeros(H)
    #initializing weight and bias matrices for final layer
    w2 = std*np.random.randn(H, K)
    b2 = np.zeros(K)

    for t in range(iterations):
        indices = np.random.choice(Ntr,batch_size)
        
        rng.shuffle(indices)# To avoid overfitting shuffle the training data set
        x=x_train[indices]
        y=y_train[indices]

        #forward propagation
        h=1/(1+np.exp(-(x.dot(w1)+b1)))
        h_test=1/(1+np.exp(-((x_test).dot(w1)+b1)))
        y_pred=h.dot(w2)+b2
        y_pred_test=h_test.dot(w2)+b2
        val=y_pred_test.shape[0]
        # calculating the training and testing loss
        training_loss=(1/batch_size)*(np.square(y_pred-y)).sum()+reg*(np.sum(w1*w1)+np.sum(w2*w2))
        loss_history.append(training_loss)
        testing_loss=(1/val)*(np.square(y_pred_test-y_test)).sum()+reg*(np.sum(w1*w1)+np.sum(w2*w2))
        loss_history_test.append(testing_loss)
        
        # calculating trainning and testing accuracies
        train_accuracy=1-(1/(10*batch_size))*(np.abs(np.argmax(y,axis=1)-np.argmax(y_pred,axis=1))).sum()
        train_acc_history.append(train_accuracy)

        test_accuracy=1-(1/(10*Nte))*(np.abs(np.argmax(y_test,axis=1)-np.argmax(y_pred_test,axis=1))).sum()
        val_acc_history.append(test_accuracy)
        # Print for every 10 iterations
        if t%10 == 0:
            print('epoch %d/%d: loss= %f || , test loss= %f ||, train accuracy= %f ||, test accuracy= %f ||, learning rate= %f ||' % (t,iterations,training_loss,testing_loss,train_accuracy,test_accuracy,lr))

        # Backward propagation
        #let's find the deravatives of the learnable parameters
        dy_pred=(1./batch_size)*2.0*(y_pred-y)#partial deravative of L w.r.t y_pred
        dw2=h.T.dot(dy_pred)+reg*w2
        db2=dy_pred.sum(axis=0)
        dh=dy_pred.dot(w2.T)
        dw1=x.T.dot(dh*h*(1-h))+reg*w1
        db1=(dh*h*(1-h)).sum(axis=0)
        #update weight matrices
        w1-=lr*dw1 
        w2-=lr*dw2
        #update bias matrices
        b1-=lr*db1
        b2-=lr*db2
        lr_array.append(lr)
        lr*=lr_decay#decaying the learning rate
    return w1,b1,w2,b2,loss_history,loss_history_test,train_acc_history,val_acc_history,lr_array
x_train_2layer,y_train_2layer,x_test_2layer,y_test_2layer,K,Din,Ntr,Nte=preprocessing(normalize=False,reshape=True)
#Remove the normalization. Otherwise the model will not learn 
iterations = 300#gradient descent iterations
lr = 1.4e-2#learning rate
lr_decay= 0.999
reg = 5e-6#lambda=regularization parameter
H=200 #hidden layer nodes  
w1_2layer,b1_2layer,w2_2layer,b2_2layer,loss_history_2layer,loss_history_test_2layer,train_acc_history_2layer,val_acc_history_2layer,lr_array_2layer=layer_2(x_train_2layer,y_train_2layer,x_test_2layer,y_test_2layer,Din,lr,lr_decay,H,reg,K,Ntr,Nte)


# ploting the graphs of training and testing losses, training and testing accuracies and learning rate
fig, axes  = plt.subplots(1,5,figsize=(50,10))
titles = {"Training Loss":loss_history_2layer, "testing loss":loss_history_test_2layer,"Training Accuracy":train_acc_history_2layer,
         "testing Accuracy": val_acc_history_2layer, "Learning Rate":lr_array_2layer}
place = 0
for key in titles.keys():
    axes[place].plot(titles[key])
    axes[place].set_title(key)
    place+=1
plt.show()
#plotting the weight matrix W as 10 images
images=[]
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']#classes of cifar-10
for i in range(w1_2layer.shape[1]):
    reshapen=np.reshape(w1_2layer[:,i]*255,(32,32,3))
    normalized=cv.normalize(reshapen, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    images.append(normalized)
fig,ax=plt.subplots(2,5,figsize=(40,10))
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(images[i*5+j],vmin=0,vmax=255)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_title(classes[i*5+j])
plt.show()

#part 3

# Function for two layer dense network with stochastic gradient descent
def mini_batching(x_train,y_train,x_test,y_test,Din,lr,lr_decay,H,reg,K,Ntr,Nte):
    loss_history = []
    loss_history_test = []
    train_acc_history = []
    val_acc_history = []
    lr_array =[]
    seed = 0
    rng = np.random.default_rng(seed=seed)
    batch_size=500 #make batch size =500 for stochastic gradient descent

    std=1e-5
    #initializing weight and bias matrices for hidden layer
    w1 = std*np.random.randn(Din, H)
    b1 = np.zeros(H)
    #initializing weight and bias matrices for final layer
    w2 = std*np.random.randn(H, K)
    b2 = np.zeros(K)

    for t in range(iterations):
        indices = np.random.choice(Ntr,batch_size)
        
        rng.shuffle(indices)# To avoid overfitting shuffle the training data set
        x=x_train[indices]
        y=y_train[indices]

        #forward propagation
        h=1/(1+np.exp(-(x.dot(w1)+b1)))
        y_pred=h.dot(w2)+b2
        h_test=1/(1+np.exp(-((x_test).dot(w1)+b1)))
        y_pred_test=h_test.dot(w2)+b2
        val=y_pred_test.shape[0]
        # calculating the training and testing loss
        training_loss=(1/batch_size)*(np.square(y_pred-y)).sum()+reg*(np.sum(w1*w1)+np.sum(w2*w2))
        loss_history.append(training_loss)
        testing_loss=(1/val)*(np.square(y_pred_test-y_test)).sum()+reg*(np.sum(w1*w1)+np.sum(w2*w2))
        loss_history_test.append(testing_loss)
        # calculating trainning and testing accuracies
        train_accuracy=1-(1/(10*batch_size))*(np.abs(np.argmax(y,axis=1)-np.argmax(y_pred,axis=1))).sum()
        train_acc_history.append(train_accuracy)

        test_accuracy=1-(1/(10*Nte))*(np.abs(np.argmax(y_test,axis=1)-np.argmax(y_pred_test,axis=1))).sum()
        val_acc_history.append(test_accuracy)
        # Print for every 10 iterations
        if t%10 == 0:
            print('epoch %d/%d: loss= %f || , test loss= %f ||, train accuracy= %f ||, test accuracy= %f ||, learning rate= %f ||'  % (t,iterations,training_loss,testing_loss,train_accuracy,test_accuracy,lr))

        # Backward propagation
        #let's find the deravatives of the learnable parameters
        dy_pred=(1./batch_size)*2.0*(y_pred-y)#partial deravative of L w.r.t y_pred
        dw2=h.T.dot(dy_pred)+reg*w2
        db2=dy_pred.sum(axis=0)
        dh=dy_pred.dot(w2.T)
        dw1=x.T.dot(dh*h*(1-h))+reg*w1
        db1=(dh*h*(1-h)).sum(axis=0)
        #update weight matrices
        w1-=lr*dw1 
        w2-=lr*dw2
        #update bias matrices
        b1-=lr*db1
        b2-=lr*db2
        lr_array.append(lr)
        lr*=lr_decay#decaying the learning rate
    return w1,b1,w2,b2,loss_history,loss_history_test,train_acc_history,val_acc_history,lr_array


batch_size = 500
H=200#hidden layer nodes  
iterations = 300#gradient descent iterations
lr = 1.4e-2
lr_decay= 0.999
reg = 5e-6#lambda=regularization parameter
x_train_2layer,y_train_2layer,x_test_2layer,y_test_2layer,K,Din,Ntr,Nte=preprocessing(normalize=False,reshape=True)
#Remove the normalization. Otherwise the model will not learn 
w1_batching,b1_batching,w2_batching,b2_batching,loss_history_batching,loss_history_test_batching,train_acc_history_batching,val_acc_history_batching,lr_array_batching=mini_batching(x_train_2layer,y_train_2layer,x_test_2layer,y_test_2layer,Din,lr,lr_decay,H,reg,K,Ntr,Nte)

# ploting the graphs of training and testing losses, training and testing accuracies and learning rate
fig, axes  = plt.subplots(1,5,figsize=(50,10))
titles = {"Training Loss":[loss_history_2layer,loss_history_batching], "testing loss":[loss_history_test_2layer,loss_history_test_batching],
"Training Accuracy":[train_acc_history_2layer,train_acc_history_batching], "testing Accuracy": [val_acc_history_2layer,val_acc_history_batching], 
"Learning Rate":[lr_array_2layer,lr_array_batching]}
place = 0
for key in titles.keys():
    if place==0:
        axes[place].plot(titles[key][0],label='with gradient descent')
        axes[place].plot(titles[key][1],label='with stochastic gradient descent')
        axes[place].legend()
    else:
        axes[place].plot(titles[key][0])
        axes[place].plot(titles[key][1])
    axes[place].set_xlabel("epoch")
    axes[place].set_title(key)
    place+=1
plt.show()

#part 4

