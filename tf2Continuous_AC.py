import sys
sys.path.insert(0, 'Utilities')
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow_probability as tfp
import numpy

tf.keras.backend.set_floatx('float32')
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, tb, X_f, layers, lb, ub):

        self.r_seed = 1234
        self.random_seed(seed=1234)
        
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.lb = lb
        self.ub = ub
                      
        self.x0 = (X0[:,0:1]).astype(np.float32)
        self.t0 = (X0[:,1:2]).astype(np.float32)

        self.x_lb = (X_lb[:,0:1]).astype(np.float32)
        self.t_lb = (X_lb[:,1:2]).astype(np.float32)

        self.x_ub = (X_ub[:,0:1]).astype(np.float32)
        self.t_ub = (X_ub[:,1:2]).astype(np.float32)
        
        self.x_f = (X_f[:,0:1]).astype(np.float32)
        self.t_f = (X_f[:,1:2]).astype(np.float32)
        
        self.u0 = (u0).astype(np.float32)

        self.activ="tanh"
        self.w_init = "glorot_normal"
        self.b_init ="zeros"
        self.data_type = tf.float32
        self.layers = layers

        self.dnn = self.dnn_init(self.layers)
        self.params = self.dnn.trainable_variables

        self.optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def dnn_init(self, layers):
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.InputLayer(layers[0]))
        network.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))

        for l in range(len(layers) - 2):
            network.add(tf.keras.layers.Dense(layers[l+1], activation=self.activ, use_bias=True,
                                              kernel_initializer=self.w_init, bias_initializer=self.b_init,
                                              kernel_regularizer=None, bias_regularizer=None,
                                              activity_regularizer=None, kernel_constraint=None,
                                              bias_constraint=None))

        network.add(tf.keras.layers.Dense(layers[len(layers) - 1]))
        return network

    def net_u(self, x, t):
        t = tf.convert_to_tensor(t, dtype=self.data_type)
        x = tf.convert_to_tensor(x, dtype=self.data_type)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(t)
            tp.watch(x)

            u = self.dnn(tf.concat([x, t], 1))
        u_x = tp.gradient(u, x)
        del tp

        return u, u_x

    def net_f(self, x, t):
        t = tf.convert_to_tensor(t, dtype=self.data_type)
        x = tf.convert_to_tensor(x, dtype=self.data_type)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(t)
            tp.watch(x)

            u = self.dnn(tf.concat([x, t], 1))
            u_x = tp.gradient(u, x)
        u_t = tp.gradient(u, t)
        u_xx = tp.gradient(u_x, x)
        del tp

        f = 5.0*u - 5.0*u**3 + 0.0001*u_xx - u_t #####
        return f


    @tf.function
    def loss_glb(self,x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f):
        u0_pred, _ = self.net_u(x_0, t_0)
        u_lb_pred, u_x_lb_pred = self.net_u(x_l, t_l)
        u_ub_pred, u_x_ub_pred = self.net_u(x_u, t_u)
        loss_prd = tf.reduce_mean(tf.square(u_0 - u0_pred)) + \
                   tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
                   tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred))
        loss_prd = 20*loss_prd
        f_pred = self.net_f(x_f, t_f)
        loss_pde = tf.reduce_mean(tf.square(f_pred))
        loss_glb = loss_prd+ loss_pde
        return loss_glb

    def loss_grad(self,x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f):
        with tf.GradientTape(persistent=True) as tp:
            loss = self.loss_glb(x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad

    @tf.function
    def grad_desc(self,x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f):
        loss, grad = self.loss_grad(x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss


    ###L - BFGS solver
    def function_factory(self,x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.

        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.params)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = numpy.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.params[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
               params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = self.loss_glb(x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value,self.params)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            f.iter.assign_add(1)
            tf.print("Iter:", f.iter, "loss:", loss_value)

            # store loss value so we can retrieve later
            tf.py_function(f.history.append, inp=[loss_value], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []

        return f

        
    def train(self, nIter):

        x_0=self.x0
        t_0=self.t0

        u_0=self.u0

        x_l=self.x_lb
        t_l=self.t_lb

        x_u=self.x_ub
        t_u=self.t_ub

        x_f=self.x_f
        t_f=self.t_f
        start_time = time.time()
        for it in range(nIter):
            loss_value=self.grad_desc(x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f)
            
            #Print
            if it % 1 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        func = self.function_factory( x_0,t_0,u_0,x_l,t_l,x_u,t_u,x_f,t_f)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.params)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=10000,
        num_correction_pairs=50,f_relative_tolerance=1.0 * np.finfo(float).eps)
        # lbfgs(loss_and_flat_grad,
        #       get_weights(u_model),
        #       Struct(), maxIter=newton_iter, learningRate=0.8)
    
    def predict(self, X_star):

        t = X_star[:,1:2]
        x =X_star[:,0:1]
        u_star, _ = self.net_u(x, t)
        f_star = self.net_f(x, t)
        return u_star,  f_star
    
if __name__ == "__main__":
    np.random.seed(1234)
    noise = 0.0      

    # Domain bounds
    lb = np.array([-1.0, 0.0]) #x,t
    ub = np.array([1.0, 1.0]) #x,t #####################

    N0 = 512
    N_b = 100
    N_f = 20000
    layers = [2, 128, 128, 128, 128, 1]
    
    data = scipy.io.loadmat('Data/AC.mat') ##########newdata_01t.mat

    t = data['tt'].flatten()[:,None] #(1，201) (1,21)
    x = data['x'].flatten()[:,None] #(1，512)
    Exact = data['uu'] ##########

    X, T = np.meshgrid(x,t) 
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) 
    u_star = Exact.T.flatten()[:,None]      ###########      

    idx_x = np.random.choice(x.shape[0], N0, replace=False) ####replace=False
    x0 = x[idx_x,:]
    u0 = Exact[idx_x,0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False) ###replace=False
    tb = t[idx_t,:]

    X_f = lb + (ub-lb)*lhs(2, N_f)
        
    model = PhysicsInformedNN(x0, u0, tb, X_f, layers, lb, ub)

    countnum = 1
    start_time = time.time()                
    model.train(10000) #10000


    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred, f_pred = model.predict(X_star)
    u_pred=u_pred.numpy()
    f_pred = f_pred.numpy()
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    F_pred = griddata(X_star, f_pred.flatten(), (X, T), method='cubic')


    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')

    ######## Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.1, right=0.9, wspace=0)  ###
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[50] * np.ones((2, 1)), line, 'k--', linewidth=1) ####################
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1) ######################
    ax.plot(t[150] * np.ones((2, 1)), line, 'k--', linewidth=1) ######################3

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|u(t,x)|_Adam$', fontsize=10)

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.8)  ####
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[:, 50], 'b-', linewidth=2, label='Exact') #################
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction') ################
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|u(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[50]), fontsize=10) ##################
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])  ###################
    ax.set_ylim([-1.1, 0.5])  #####################

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[:, 100], 'b-', linewidth=2, label='Exact') ########################
    ax.plot(x, U_pred[100, :], 'r--', linewidth=2, label='Prediction') ######################
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|u(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])  #########################
    ax.set_ylim([-1.1, 0.5])  #################
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10) ####################3
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[:, 150], 'b-', linewidth=2, label='Exact') ##################
    ax.plot(x, U_pred[150, :], 'r--', linewidth=2, label='Prediction') ###################
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|u(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])  ####################
    ax.set_ylim([-1.1, 0.5])  ####################
    ax.set_title('$t = %.2f$' % (t[150]), fontsize=10) #################

    plt.show()



