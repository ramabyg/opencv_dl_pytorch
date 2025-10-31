import torch
import matplotlib.pyplot as plt
from collections import defaultdict

plt.style.use('ggplot')
torch.manual_seed(0)

# Generating y = mx + c + random noise
num_data = 1000

# True values of m and c
m_line = 3.3
c_line = 5.3

# input (Generate random data between [-5,5])
x = 10 * torch.rand(num_data) - 5

# Output (Generate data assuming y = mx + c + noise)
y_label = m_line * x + c_line + torch.randn_like(x)

# Add a few outlier
num_outliers = int(0.05 * num_data)
random_index = torch.randint(num_data, (num_outliers,))
y_label[random_index] = 50 * torch.rand(len(random_index))

y = m_line * x + c_line

# Plot the generated data points
plt.plot(x, y_label, '.', color='g', label="Data points")
plt.plot(x, y, color='b', label='y = mx + c', linewidth=3)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()


def MSE_loss(inputs, label, m, c):
    '''
    All arguments are defined in the training section of this notebook.
    This function will be called from the training section.
    So before completing this function go through the whole notebook.

    inputs (torch.tensor): input (X)
    labels (torch.tensor): label (Y)
    m (float): slope of the line
    c (float): vertical intercept of line
    '''

    # Mean square error (loss)
    #loss = None
    predictions = m * inputs + c
#     mse_loss = torch.nn.MSELoss()
#     loss = mse_loss(label, predictions)
    loss = torch.mean((label-predictions) ** 2)



    ###
    ### YOUR CODE HERE
    ###


    return loss

def MSE_loss(inputs, label, m, c):

        # Mean square error (loss)
        #loss = None
        predictions = m * inputs + c
    #     mse_loss = torch.nn.MSELoss()
    #     loss = mse_loss(label, predictions)
        loss = torch.mean((label-predictions) ** 2)



        ###
        ### YOUR CODE HERE
        ###


        return loss

def MAE_loss(inputs, label, m, c):
    # Mean absolute error (loss)
    loss = None
    predictions = m * inputs + c
    loss = torch.mean(torch.abs(label - predictions))

    ###
    ### YOUR CODE HERE
    ###

    return loss


def gradient_wrt_m_and_c_mse(inputs, labels, m, c):

    '''
    All arguments are defined in the training section of this notebook.
    This function will be called from the training section.
    So before completing this function go through the whole notebook.

    inputs (torch.tensor): input (X)
    labels (torch.tensor): label (Y)
    m (float): slope of the line
    c (float): vertical intercept of line
    '''
    # gradient w.r.t to m is g_m
    g_m = None

    # gradient w.r.t to c is g_c
    g_c = None

    predictions = m * inputs + c
    loss = labels - predictions
    signs = torch.sign(loss)
#     g_m = -1 * torch.mean(signs * inputs)
#     g_c = -1 * torch.mean(signs)
    g_m = -2 * torch.mean(inputs * loss)
    g_c = -2 * torch.mean(loss)

    ###
    ### YOUR CODE HERE
    ###

    return g_m, g_c

def gradient_wrt_m_and_c_mae(inputs, labels, m, c):

    '''
    All arguments are defined in the training section of this notebook.
    This function will be called from the training section.
    So before completing this function go through the whole notebook.

    inputs (torch.tensor): input (X)
    labels (torch.tensor): label (Y)
    m (float): slope of the line
    c (float): vertical intercept of line
    '''

    # gradient w.r.t to m is g_m
    g_m = None

    # gradient w.r.t to c is g_c
    g_c = None
    predictions = m * inputs + c
    loss = labels - predictions
    signs = torch.sign(loss)
    g_m = -1 * torch.mean(signs * inputs)
    g_c = -1 * torch.mean(signs)

    ###
    ### YOUR CODE HERE
    ###

    return g_m, g_c

def update_m_and_c(m, c, g_m, g_c, lr):
    '''
    All arguments are defined in the training section of this notebook.
    This function will be called from the training section.
    So before completing this function go through the whole notebook.

    g_m = gradient w.r.t to m
    c_m = gradient w.r.t to c
    '''
    updated_m = m -  lr * g_m
    updated_c = c - lr * g_c

    return updated_m, updated_c

def plot_loss(loss):
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss[0])), loss[0], color='k')

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('MSE Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss[1])), loss[1], color='r')

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('MAE Loss')
    plt.show()

    return

def display_training(X, Y_origin, Y_label, loss, m, c, iteration):
    print('Iteration: {}, \nLoss_mse: {:.3f}, m_mse: {:.3f}, c_mse: {:.3f}\nLoss_mae: {:.3f}, m_mae: {:.3f},'
          'c_mae: {:.3f}'.format(iteration, loss[0][-1], m[0], c[0], loss[1][-1], m[1], c[1]))

    # Prediction for trained with MSE loss
    y_pred_mse = m[0] * X + c[0]

    # Prediction for trained with MAE loss
    y_pred_mae = m[1] * X + c[1]

    # plots

    # points plot
    plt.plot(X, Y_label, '.', color='g')

    # Line for which data is generated
    plt.plot(X, Y_origin, color='b', label='Line corresponding to m={0:.2f}, c={1:.2f}'.
             format(m_line, c_line), linewidth=3)

    # Line learned with MSE loss
    plt.plot(X, y_pred_mse, color='k', label='Line corresponding to m_mse={0:.2f}, c_learned={1:.2f}'.
             format(m[0], c[0]), linewidth=3)

    # Line learned with MSE loss
    plt.plot(X, y_pred_mae, color='r', label='Line corresponding to m_mae={0:.2f}, c_learned={1:.2f}'.
             format(m[1], c[1]), linewidth=3)


    plt.title("Iteration : {}".format(iteration))
    plt.legend()

    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    return

def train(inputs, labels, labels_origin, initial_m, initial_c, grad_fun_m_c_list, loss_fun_list, lr=0.01,
          batch_size=10, epoch=10, display_count=20):

    loss = dict()
    m = dict()
    c = dict()

    for i in range(len(grad_fun_m_c_list)):
        loss[i] = []
        m[i] = initial_m
        c[i] = initial_c

    num_batches = int(len(inputs)/batch_size)

    for i in range(epoch):

        shuffle_indices = torch.randint(0, len(inputs), (len(inputs),))

        for j in range(num_batches):

            X = inputs[shuffle_indices[j*batch_size:j*batch_size+batch_size]]
            Y = labels[shuffle_indices[j*batch_size:j*batch_size+batch_size]]

            for k, grad_m_c in enumerate(grad_fun_m_c_list):
                g_m, g_c = grad_m_c(X, Y, m[k], c[k])

                m[k], c[k] = update_m_and_c(m[k], c[k], g_m, g_c, lr)
                l = loss_fun_list[k](inputs, labels, m[k], c[k])
                loss[k].append(l)

            if j % display_count == 0:
                iteration = i * num_batches + j
                display_training(inputs, labels_origin, labels, loss, m, c, iteration)

    final_iteration = (epoch-1) * num_batches + num_batches - 1

    return m, c, loss, final_iteration


# inputs
inputs = x

# output/labels
labels = y_label

# labels around y
labels_origin = y

# epoch
epoch=20

# learning rate
lr = 0.005

# batch size
batch_size=10

# dislpay plot count
display_count=40

# inital m
initial_m = 2

# initail c
initial_c = 1

grad_fun_m_c_list = [gradient_wrt_m_and_c_mse, gradient_wrt_m_and_c_mae]

loss_fun_list = [MSE_loss, MAE_loss]

m, c, loss, final_iteration = train(inputs, labels, labels_origin, initial_m, initial_c, grad_fun_m_c_list,
                                    loss_fun_list, lr, batch_size, epoch, display_count)


print('{0}\nFinal plots\n{0}'.format('--------------------------'))

display_training(inputs, labels_origin, labels, loss, m, c, iteration=final_iteration)

plot_loss(loss)

