import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Defining function for data preprocessing
def preprocessing():
    # Importing data set CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    K = len(np.unique(y_train)) # no.of Classes 
    Ntr = x_train.shape[0] # Number of training data
    Nte = x_test.shape[0] # Number of testing data
    Din = 3072 # CIFAR10 = 3072 = 32x32x3

    x_train, x_test = x_train / 255.0, x_test / 255.0# Normalize pixel values

    # Center the pixel values
    mean_image = np.mean(x_train, axis=0)
    x_train = x_train - mean_image
    x_test = x_test - mean_image
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=K)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=K)

    #flatterning the input images and changing the data type
    
    x_train = np.reshape(x_train,(Ntr,Din))
    x_test = np.reshape(x_test,(Nte,Din))
    
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

        # forward pass
        y_pred=x.dot(w1)+b1
        y_pred_test=x_test.dot(w1)+b1

        # calculating loss using regularized loss function
        train_loss=(1/batch_size)*(np.square(y_pred-y)).sum()+reg*(np.sum(w1*w1))
        loss_history.append(train_loss)
        test_loss=(1/batch_size)*(np.square(y_pred_test-y_test)).sum()+reg*(np.sum(w1*w1))
        loss_history_testing.append(test_loss)
        
        # calculating trainning and testing accuracies
        train_accuracy=1-(1/(batch_size*K))*(np.abs(np.argmax(y,axis=1)-np.argmax(y_pred,axis=1))).sum()
        train_acc_history.append(train_accuracy)

        test_accuracy=1-(1/(batch_size*K))*(np.abs(np.argmax(y_test,axis=1)-np.argmax(y_pred_test,axis=1))).sum()
        val_acc_history.append(test_accuracy)

        if t%10 == 0:
            print('epoch %d/%d: train loss= %f || ,test loss= %f ||,train accracy= %f ||, test accracy= %f ||, learning rate= %f ||' % (t,iterations,train_loss,test_loss,train_accuracy,test_accuracy,lr))

        # Backward pass
        dy_pred=(1./batch_size)*2.0*(y_pred-y)#partial deravative of L w.r.t y_pred
        dw1=x.T.dot(dy_pred)+reg*w1
        db1=dy_pred.sum(axis=0)

        # updating parameters
        w1-=lr*dw1#update weight matrix
        b1-=lr*db1#update bias matrix
        lr_array.append(lr)
        lr*=lr_decay#decaying learning rate
    return w1,b1,loss_history,loss_history_testing,train_acc_history,val_acc_history,lr_array


#defining parameters and running the linear classifier 
iterations = 300#gradient descent iterations
lr = 1.4e-2#learning rate
lr_decay= 0.999
reg = 5e-6 #lamda(regularization constant for the loss function)
x_train,y_train,x_test,y_test,K,Din,Ntr,Nte=preprocessing()
w1,b1,loss_history,loss_history_test,train_acc_history,val_acc_history,lr_array=layer1LinearClassifier(x_train,y_train,x_test,y_test,K,Din,lr,lr_decay,reg,Ntr,Nte)

# ploting the graphs of trining and testing losses, training and testing accuracies and learning rate
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
