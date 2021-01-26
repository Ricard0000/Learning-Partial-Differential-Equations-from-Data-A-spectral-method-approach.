#

import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io
from scipy.interpolate import griddata
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat

def Derivative_of_data(x,Nx,n,Data,stencil_order):
    h=x[0,2]-x[0,1]
    if stencil_order==2:
        xx=x[0,1:Nx-1]
        U=Data[1:Nx-1,:]
        Dx=(Data[2:Nx,:]-Data[0:Nx-2,:])/(2.0*h)
        Dxx=(Data[2:Nx,:]-2.0*Data[1:Nx-1,:]+Data[0:Nx-2,:])/(1.0*h*h)
    elif stencil_order==4:
        xx=x[0,2:Nx-2]
        U=Data[2:Nx-2,:]
        Dx=(Data[0:Nx-4,:]-8.0*Data[1:Nx-3,:]+8.0*Data[3:Nx-1,:]-Data[4:Nx,:])/(12.0*h)
        Dxx=(-Data[0:Nx-4,:]+16.0*Data[1:Nx-3,:]-30.0*Data[2:Nx-2,:]+16.0*Data[3:Nx-1,:]-Data[4:Nx,:])/(12.0*h*h)
    else:
        xx=x[0,3:Nx-3]
        U=Data[3:Nx-3,:]
        Dx=(-1.0*Data[0:Nx-6,:]+9.0*Data[1:Nx-5,:]-45.0*Data[2:Nx-4,:]+45.0*Data[4:Nx-2,:]-9.0*Data[5:Nx-1,:]+1.0*Data[6:Nx,:])/(60.0*h)
        Dxx=(2.0*Data[0:Nx-6,:]-27.0*Data[1:Nx-5,:]+270.0*Data[2:Nx-4,:]-490.0*Data[3:Nx-3,:]+270.0*Data[4:Nx-2,:]-27.0*Data[5:Nx-1,:]+2.0*Data[6:Nx,:])/(180.0*h*h)
    return xx,U,Dx,Dxx


#This is to compute solutions after training is finished
def The_Kernel(u, Dx,Dxx, weights, biases,N_layers,dt,Nx):
    H=u
    F_rhs=0.0*H
    Z=0.0*u
    print(Z.shape)
    o=np.ones([Nx,1])
    print(Dx.shape)
    print(Dxx.shape)
    Input=[]
    Input.append(Z)
    Input.append(Dx)
    Input.append(Dxx)
    for J in range(0,N_layers):
        W=weights[J]
        b=biases[J]
        list11=[]
        list12=[]
        for I in range(0,J+3):
            list11.append(W[0,I]*Input[I])
            list12.append(W[1,I]*Input[I])
        xi1=sum(list11)+b[0,0]*o
        xi2=sum(list12)+b[1,0]*o
#        print(xi1.shape)
        f=np.multiply(xi1,xi2)
        Input.append(f)
    W=weights[N_layers]
    b=biases[N_layers]
    for I in range(0,N_layers+3):
        F_rhs=F_rhs+W[0,I]*Input[I] 
    F_rhs=F_rhs+b
    U_Predict=np.multiply(o,np.exp(-dt*F_rhs))
    return U_Predict, F_rhs


class PDE_NET_Spectral:
    # Initialize the class
    def __init__(self, u0r,u0i, K, K2, K3, K4, u_datar,u_datai, n,N_sp,N_layers,Nx,dt,Losss,The_solutionr,The_solutioni,u0,u0x,u0xx,u00,u00x,u00xx):

        self.u0r = u0r
        self.u0i = u0i
        self.u_datar = u_datar
        self.u_datai = u_datai
        self.K = K
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.Losss=Losss
        self.The_solutionr=The_solutionr
        self.The_solutioni=The_solutioni
        self.u0 = u0
        self.u0x = u0x
        self.u0xx = u0xx
        self.u00 = u00
        self.u00x = u00x
        self.u00xx = u00xx
        
        
        
        # Initialize NNs
        self.Nx=Nx
        self.N_layers=N_layers
        self.N_sp=N_sp
        self.n=n
        self.sp_weightsr,self.sp_weightsi,self.weights,self.biases = self.initialize_NN(N_sp,N_layers)
        
        # tf Placeholders        
        self.u0r_tf = tf.placeholder(tf.float32, shape=[self.u0r.shape[0],self.u0r.shape[1]])
        self.u_datar_tf = tf.placeholder(tf.float32, shape=[self.u_datar.shape[0], self.u_datar.shape[1]])
        self.u0i_tf = tf.placeholder(tf.float32, shape=[self.u0i.shape[0],self.u0i.shape[1]])
        self.u_datai_tf = tf.placeholder(tf.float32, shape=[self.u_datai.shape[0], self.u_datai.shape[1]])
        self.K_tf = tf.placeholder(tf.float32, shape=[self.K.shape[0], self.K.shape[1]])
        self.K2_tf = tf.placeholder(tf.float32, shape=[self.K2.shape[0], self.K2.shape[1]])
        self.K3_tf = tf.placeholder(tf.float32, shape=[self.K3.shape[0], self.K3.shape[1]])
        self.K4_tf = tf.placeholder(tf.float32, shape=[self.K4.shape[0], self.K4.shape[1]])
        self.The_solutionr_tf = tf.placeholder(tf.float32, shape=[self.The_solutionr.shape[0], self.The_solutionr.shape[1]])
        self.The_solutioni_tf = tf.placeholder(tf.float32, shape=[self.The_solutioni.shape[0], self.The_solutioni.shape[1]])
                
        self.u0_tf = tf.placeholder(tf.float32, shape=[self.u0.shape[0],self.u0.shape[1]])
        self.u0x_tf = tf.placeholder(tf.float32, shape=[self.u0x.shape[0],self.u0x.shape[1]])
        self.u0xx_tf = tf.placeholder(tf.float32, shape=[self.u0xx.shape[0],self.u0xx.shape[1]])
        
        self.u00_tf = tf.placeholder(tf.float32, shape=[self.u00.shape[0],self.u00.shape[1]])
        self.u00x_tf = tf.placeholder(tf.float32, shape=[self.u00x.shape[0],self.u00x.shape[1]])
        self.u00xx_tf = tf.placeholder(tf.float32, shape=[self.u00xx.shape[0],self.u00xx.shape[1]])
        
        
        # tf Graphs
        self.u0r_pred,self.u0i_pred,self.u_predr,self.u_predi,self.sp_weightsr_tf,self.sp_weightsi_tf,self.weights_tf, self.b_tf = self.net_uv(self.u0r_tf,self.u0i_tf,self.K_tf,self.K2_tf,self.K3_tf,self.K4_tf,self.u0_tf,self.u0x_tf,self.u0xx_tf,self.u00_tf,self.u00x_tf,self.u00xx_tf)

        # Loss


#        self.loss = tf.reduce_mean(tf.abs(self.u_datar_tf - self.u0r_pred)+tf.abs(self.u_datai_tf - self.u0i_pred))#*(tf.abs(self.The_solution_tf - self.u_pred)))
#        self.loss = tf.reduce_mean(tf.abs(self.The_solutionr_tf - self.u_predr)+tf.abs(self.The_solutioni_tf - self.u_predi))+0.00000001*(tf.reduce_mean(tf.abs(self.weights_tf[0]))+tf.reduce_mean(tf.abs(self.weights_tf[1]))+tf.reduce_mean(tf.abs(self.weights_tf[2]))+tf.reduce_mean(tf.abs(self.weights_tf[3]))+tf.reduce_mean(tf.abs(self.b_tf[0]))+tf.reduce_mean(tf.abs(self.b_tf[1]))+tf.reduce_mean(tf.abs(self.b_tf[2]))+tf.reduce_mean(tf.abs(self.b_tf[3]))+tf.reduce_mean(tf.abs(self.sp_weightsr_tf[0]))+tf.reduce_mean(tf.abs(self.sp_weightsi_tf[0])))
        #Try 0.00001 for sparsity
        self.loss = tf.reduce_mean(tf.abs(self.The_solutionr_tf - self.u_predr)+tf.abs(self.The_solutioni_tf - self.u_predi))+0.00001*(tf.reduce_mean(tf.abs(self.weights_tf[N_layers]))+tf.reduce_mean(tf.abs(self.weights_tf[N_layers-1]))+tf.reduce_mean(tf.abs(self.weights_tf[N_layers-2]))+tf.reduce_mean(tf.abs(self.weights_tf[N_layers-3]))+tf.reduce_mean(tf.abs(self.sp_weightsr_tf[0]))+tf.reduce_mean(tf.abs(self.sp_weightsi_tf[0])))
        


#        self.loss = tf.reduce_mean(tf.abs(self.u_datar_tf - self.u0r_pred)+tf.abs(self.u_datai_tf - self.u0i_pred))
#        self.loss = tf.reduce_mean(tf.square(self.The_solution_tf - self.u0r_pred))
        # Optimizers
                # Optimizers
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

#        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
#                                                                method = 'L-BFGS-B', 
#                                                                options = {'maxiter': 50000,
#                                                                           'maxfun': 50000,
#                                                                           'maxcor': 50,
#                                                                           'maxls': 50,
#                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self,N_sp,N_layers):
        sp_weightsr=[]
        sp_weightsi=[]
        weights=[]
        biases=[]
        for I in range(1,N_layers+1):
            W=0.05*tf.Variable(tf.ones([2,2+I],dtype=tf.float32),dtype=tf.float32)
            b=0.05*tf.Variable(tf.ones([2,1], dtype=tf.float32),dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        W_last=0.05*tf.Variable(tf.ones([1,3+N_layers], dtype=tf.float32), dtype=tf.float32)
        bias_last=tf.Variable(0.05, dtype=tf.float32)
        weights.append(W_last)
        biases.append(bias_last)
        W=0.05*tf.Variable(tf.ones([1,N_sp],dtype=tf.float32),dtype=tf.float32)
        sp_weightsr.append(W)
        W=0.05*tf.Variable(tf.ones([1,N_sp],dtype=tf.float32),dtype=tf.float32)
        sp_weightsi.append(W)
        return sp_weightsr,sp_weightsi,weights, biases
    
    
    #5
    
    def neural_net(self, Xr,Xi, K,K2,K3,K4,sp_weightsr,sp_weightsi,weights,biases,n,N_sp,N_layers,Nx,dt,u0,u0x,u0xx,u00,u00x,u00xx):
     #   H=Xr
        F_rhs1=0.0*Xr
        F_rhs2=0.0*Xr
        Input=[]

        Input.append(u0)
        Input.append(u0x)
        Input.append(u0xx)
        for J in range(0,N_layers):
            print('1')
            W=weights[J]
            b=biases[J]
            list11=[]
            list12=[]
            for I in range(0,J+3):
                list11.append(W[0,I]*Input[I])
                list12.append(W[1,I]*Input[I])
            xi1=sum(list11)+b[0,0]
            xi2=sum(list12)#+b[1,0]
            f=tf.multiply(xi1,xi2)
            Input.append(f)
        W=weights[N_layers]
#        b=biases[N_layers]
        for I in range(3,N_layers+3):
            print('2')
            F_rhs1=F_rhs1+W[0,I]*Input[I]

#        F_rhs1=W[0,0]*Input[0]+W[0,1]*Input[1]+W[0,2]*Input[2]
            
        print('3')
        uastr=tf.multiply(tf.math.exp(-1.0*F_rhs1*dt/2.0),Xr)
        
        uasti=tf.multiply(tf.math.exp(-1.0*F_rhs1*dt/2.0),Xi)
        
        
#        uastr=tf.math.real(uastr)
#        uasti=tf.math.imag(uasti)
        
        Temp1r=tf.cast(uastr, tf.complex64)
        Temp1i=tf.cast(uasti, tf.complex64)
        
        uast=tf.signal.fft(Temp1r+1j*Temp1i)
        Xrr=tf.math.real(uast)
        Xii=tf.math.imag(uast)
        
        
        
        W_spr=sp_weightsr[0]
        W_spi=sp_weightsi[0]
        
        if N_sp==2:
#            F_rhs=W_sp[0,0]*K+W_sp[0,1]*K2
            U_Predictr=tf.multiply(Xrr,tf.math.cos(-dt*W_spi[0,0]*K-dt*W_spi[0,1]*K2))-tf.multiply(Xii,tf.math.sin(-dt*W_spi[0,0]*K-dt*W_spi[0,1]*K2))
            U_Predicti=tf.multiply(Xrr,tf.math.sin(-dt*W_spi[0,0]*K-dt*W_spi[0,1]*K2))+tf.multiply(Xii,tf.math.cos(-dt*W_spi[0,0]*K-dt*W_spi[0,1]*K2))
            U_Predictr=tf.multiply(tf.math.exp(-dt*W_spr[0,1]*K2-dt*W_spr[0,0]*K),U_Predictr)
            U_Predicti=tf.multiply(tf.math.exp(-dt*W_spr[0,1]*K2-dt*W_spr[0,0]*K),U_Predicti)
        else:
            #im=-dt*W_spr[0,0]*K-dt*W_spi[0,1]*K2+dt*W_spr[0,2]*K3+dt*W_spi[0,3]*K4
            #rm=dt*W_spi[0,0]*K-dt*W_spr[0,1]*K2-dt*W_spi[0,2]*K3+dt*W_spr[0,3]*K4
            
            im=-dt*W_spr[0,0]*K+dt*W_spr[0,2]*K3
            rm=-dt*W_spr[0,1]*K2-dt*W_spr[0,3]*K4
            
            U_Predictr=tf.multiply(Xrr,tf.math.cos(im))-tf.multiply(Xii,tf.math.sin(im))
            #U_Predicti=tf.multiply(Xrr,tf.math.sin(im))+tf.multiply(Xii,tf.math.cos(im))
            U_Predicti=tf.multiply(Xrr,tf.math.sin(im))+tf.multiply(Xii,tf.math.cos(im))
            U_Predictr=tf.multiply(tf.math.exp(rm),U_Predictr)
            U_Predicti=tf.multiply(tf.math.exp(rm),U_Predicti)
            
            
        Temp1r=tf.cast(U_Predictr, tf.complex64)
        Temp1i=tf.cast(U_Predicti, tf.complex64)
        
        uastast=tf.signal.ifft(Temp1r+1j*Temp1i)
        uastastr=tf.math.real(uastast)
        uastasti=tf.math.imag(uastast)
        
        
        
        
        
        
        
        Input2=[]
        Input2.append(u00)
        Input2.append(u00x)
        Input2.append(u00xx)
        for J in range(0,N_layers):
            print('1')
            W=weights[J]
            b=biases[J]
            list11=[]
            list12=[]
            for I in range(0,J+3):
                list11.append(W[0,I]*Input2[I])
                list12.append(W[1,I]*Input2[I])
            xi1=sum(list11)+b[0,0]
            xi2=sum(list12)#+b[1,0]
            f=tf.multiply(xi1,xi2)
            Input2.append(f)
        W=weights[N_layers]
#        b=biases[N_layers]
        for I in range(3,N_layers+3):
            print('2')
            F_rhs2=F_rhs2+W[0,I]*Input2[I]
        
        
        
        
        
        
        
        
        
        
#        F_rhs2=W[0,0]*u00+W[0,1]*u00x+W[0,2]*u00xx
        
        
        uastastr=tf.multiply(uastastr,tf.math.exp(-1.0*F_rhs2*dt/2.0))
        
        
        ur=tf.cast(uastastr, tf.float32)
        ui=tf.cast(uastasti, tf.float32)

        uas=tf.multiply(tf.math.exp(-0.0*F_rhs1*dt/2.0),Xr)#This does nothing
        
        uasr=tf.math.real(uas)
        uasi=tf.math.imag(uas)
        Temp1r=tf.cast(uasr, tf.complex64)
        Temp1i=tf.cast(uasi, tf.complex64)
        uas=tf.signal.fft(Temp1r+1j*Temp1i)
        U_ft_r=tf.math.real(uas)
        U_ft_i=tf.math.imag(uas)
        U_ft_r=tf.cast(U_ft_r, tf.float32)
        U_ft_i=tf.cast(U_ft_i, tf.float32)

        return U_ft_r, U_ft_i,ur,ui,sp_weightsr,sp_weightsi,weights,biases
    
    
    
    
    
    
    
    

#Temp1=tf.cast(The_solution, tf.complex64)
#fftts=tf.math.real(tf.signal.fft(Temp1))
#sess=tf.Session()
#cun=sess.run(fftts)
#plt.scatter(x_domain, cun[:,0],  color='black', linewidth=3, label='Predicted_short_term_solution')
#plt.scatter(x_domain, u_data[:,0],  color='black', linewidth=3, label='Predicted_short_term_solution')
#plt.scatter(x_domain, unn[0,:])
#thing=np.zeros([n_dat,Nx],dtype=complex)
#for I in range(0,Nx):
#    for J in range(0,n_dat):
#        thing[J,I]=The_solution[I,J]




    
    
    def net_uv(self, u0r,u0i,K,K2,K3,K4,u0,u0x,u0xx,u00,u00x,u00xx):
        ur,ui,upredr,upredi,sp_weightsr,sp_weightsi,weights,biases = self.neural_net(u0r,u0i, K, K2, K3, K4, self.sp_weightsr,self.sp_weightsi, self.weights, self.biases, self.n,self.N_sp,self.N_layers,Nx,dt,u0,u0x,u0xx,u00,u00x,u00xx)
        return ur,ui,upredr,upredi,sp_weightsr,sp_weightsi,weights,biases
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.u0r_tf: self.u0r,
                   self.u0i_tf: self.u0i,
                   self.K_tf: self.K,
                   self.K2_tf: self.K2,
                   self.K3_tf: self.K3,
                   self.K4_tf: self.K4,
                   self.u_datar_tf: self.u_datar,
                   self.u_datai_tf: self.u_datai,
                   self.The_solutionr_tf: self.The_solutionr,
                   self.The_solutioni_tf: self.The_solutioni,
                   self.u0_tf: self.u0,
                   self.u0x_tf: self.u0x,
                   self.u0xx_tf: self.u0xx,
                   self.u00_tf: self.u00,
                   self.u00x_tf: self.u00x,
                   self.u00xx_tf: self.u00xx}
        
        
        
        
        start_time = time.time()
        L=0
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            loss_value = self.sess.run(self.loss, tf_dict)
            Losss[L,0]=loss_value
            L=L+1
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

        
#        start_time = time.time()
#        for it in range(nIter):
#            self.sess.run(self.train_op_Adam, tf_dict)    
#            # Print
#            if it % 10 == 0:
#                elapsed = time.time() - start_time
#                loss_value = self.sess.run(self.loss, tf_dict)
#                print('It: %d, Loss: %.3e, Time: %.2f' % 
#                      (it, loss_value, elapsed))
#                start_time = time.time()
#        self.optimizer.minimize(self.sess,
#                                feed_dict = tf_dict,
#                                fetches = [self.loss],
#                                loss_callback = self.callback)
                                        
    def predict(self, u0r, u0i, K, K2,K3,K4,The_solutionr,The_solutioni):
        tf_dict = {self.u0r_tf: u0r,
                   self.u0i_tf: u0i,
                   self.K_tf: K,
                   self.K2_tf: K2,
                   self.K3_tf: K3,
                   self.K4_tf: K4,
                   self.The_solutionr_tf: self.The_solutionr,
                   self.The_solutioni_tf: self.The_solutioni,
                   self.u0_tf: u0,
                   self.u0x_tf: u0x,
                   self.u0xx_tf: u0xx,
                   self.u00_tf: u00,
                   self.u00x_tf: u00x,
                   self.u00xx_tf: u00xx}
        #u_star = self.sess.run(self.u0_tf, tf_dict)
        u_star = self.sess.run(self.u0r_pred, tf_dict)
        biases=self.sess.run(self.b_tf)
        weights=self.sess.run(self.weights_tf)
        sp_weightsr=self.sess.run(self.sp_weightsr_tf)
        sp_weightsi=self.sess.run(self.sp_weightsi_tf)
        return u_star,sp_weightsr,sp_weightsi,weights,biases
    
if __name__ == "__main__": 

    
    
    #Loading data
#    TTdata = scipy.io.loadmat('data_mm_periodic.mat')
#    xx = Tdata['x']
#    data_ps=TTdata['u']
#    dt=Tdata['dt']
#    h=Tdata['hx']
#    Nx,n=Data.shape
#    T_final=n*dt
    
    
#    TTdata = scipy.io.loadmat('spectralNN_data.mat')
    TTdata = scipy.io.loadmat('datakuramoto.mat')
    Tdata = scipy.io.loadmat('spectralNN_data_2.mat')
    Tkernel = scipy.io.loadmat('spectralNN_data_3.mat')
    
    
    
    
    
#    TTdata = scipy.io.loadmat('dataNLS.mat')

    TTdata = scipy.io.loadmat('datakuramoto.mat')


#savemat('datakuramoto.mat',{'x':x1, 'ur':saveur,'ui':saveui, 'deex':saveux, 'deexx':saveuxx, 'dt':dt,'a':a,'b':b})
    xx = TTdata['x']
    data_psr = TTdata['ur']
    data_psi = TTdata['ui']
    
#    data_ps=TTdata['usquared']
    deex=TTdata['deex']
    deexx=TTdata['deexx']
    
    dt=TTdata['dt']
    
    
#    ,{'x':x, 'ur':rsaveu,'ui':isaveu, 'dt':dt}
    
    
    #Loading data
#    xx = Tdata['xx']
#    data_ps = TTdata['data_ps']
#    data_psi = Tdata['data_psi_2']
#    T_kernel = Tkernel['The_Kernel']
#    dt=Tdata['dt']
#    xmin=Tdata['xmin']
#    xmax=Tdata['xmax']
    
    xmin=TTdata['a']
    xmax=TTdata['b']
    
#    kkk=Tdata['kkk']
    Nxx,n=data_psr.shape
    stencil_order=0
    Nx=Nxx-stencil_order
    M=Nx
    a=xmin
    b=xmax
    
    hx=(b-a)/(Nx-1)

#    a=-1.5
#    b=1.5
    
    
    
#    for I in range(0,Nx):
#        for J in range(0,n):
#           # data_ps[I,J]=np.exp(-(0+0.5*J/(n-1)))*np.sin(a+(b-a)*I/(Nx-1))
#            data_ps[I,J]=np.exp(-(0+0.5*J/(n-1)))*np.exp(-abs(a+(b-a)*I/(Nx-1)))
            
            


    xx=np.float32(xx)

    
    
    kk=np.zeros([1,Nx],dtype=float)
    Nx_half=round(Nx/2)
    for I in range(0,Nx_half+1):
        kk[0,I]=2.0*np.pi/(b-a)*I
    
    for I in range(0,Nx_half-1): 
        kk[0,Nx-I-1]=-2.0*np.pi/(b-a)*(I+1)
    
    
    
    
    
    
    
    N_layers=4

    x_domain=np.zeros([Nx,1], dtype=float)
    kernel_exact=np.zeros([Nx,1], dtype=float)
    
    
    
    
    
    
#    n_dat=16
#    n_dat=30
    n_dat_temp=round(8.0*n/9.0)-round(n/9.0)
    n_start=round(n/9.0)
#    n_start=0
    
    n_skip=20
    n_dat=round(n_dat_temp/n_skip)
    dt=dt*n_skip
    
    u0=np.zeros([n_dat,Nx], dtype=complex)
    Dx=np.zeros([n_dat,Nx], dtype=float)
    Dxx=np.zeros([n_dat,Nx], dtype=float)
    u_data=np.zeros([n_dat,Nx], dtype=complex)
    The_solutionr=np.zeros([n_dat,Nx], dtype=float)
    The_solutioni=np.zeros([n_dat,Nx], dtype=float)
    
    un=np.zeros([n,Nx],dtype=float)
#    for J in range(0,n):
#        for I in range(0,Nx):
#            un[J,I]=data_ps[I,J]
    
#    unn=np.fft.fft(un)
#    unn=np.real(unn)
#    for J in range(0,n_dat):
#        for I in range(0,Nx):
#            u0[J,I]=data_ps[I,n_skip*J+int(n_skip/2)+n_start]
#            u0[J,I]=data_ps[I,n_skip*J+n_start]
    
    
    
    u0x=np.zeros([n_dat,Nx], dtype=float)
    u0xx=np.zeros([n_dat,Nx], dtype=float)
    
    u00=np.zeros([n_dat,Nx], dtype=float)
    u00x=np.zeros([n_dat,Nx], dtype=float)
    u00xx=np.zeros([n_dat,Nx], dtype=float)
    
    for J in range(0,n_dat):
        for I in range(0,Nx):
#            u0[J,I]=unn[n_skip*J+n_start,I]
#            u_data[J,I]=unn[n_skip*(J+1)+n_start,I]
#            u0[J,I]=data_psi[I,n_skip*J+n_start]
#            u_data[J,I]=data_psi[I,n_skip*(J+1)+n_start]
            The_solutionr[J,I]=data_psr[I,n_skip*(J+1)+n_start]
            The_solutioni[J,I]=data_psi[I,n_skip*(J+1)+n_start]
            u0x[J,I]=deex[I,n_skip*(J+0)+n_start]
            u0xx[J,I]=deexx[I,n_skip*(J+0)+n_start]
            u00[J,I]=data_psr[I,n_skip*(J+1)+n_start]            
            u00x[J,I]=deex[I,n_skip*(J+1)+n_start]
            u00xx[J,I]=deexx[I,n_skip*(J+1)+n_start]
            
            
#    u0x=(np.roll(u0,-1,axis=0)-np.roll(u0,1,axis=0))/(2.0*hx)
#    u0xx=(np.roll(u0,-1,axis=0)-2.0*u0+np.roll(u0,1,axis=0))/(hx*hx)

            
            
            
#fft_the_solution=np.zeros([Nx,n_dat],dtype=float)
#for J in range(0,n_dat):
#    fft_the_solution[:,J]=np.real(np.fft.fft(The_solution[:,J]))
#fig = plt.figure()
#plt.scatter(K,fft_the_solution[:,10], color='black', linewidth=3, label='Loss')
#plt.show()
#fig = plt.figure()
#plt.scatter(K,u0[:,0], color='black', linewidth=3, label='Loss')
#plt.show()

            
    
    
    
    K=np.zeros([1,Nx], dtype=float)
    K2=np.zeros([1,Nx], dtype=float)
    K3=np.zeros([1,Nx], dtype=float)
    K4=np.zeros([1,Nx], dtype=float)
    Longrun_test_0=np.zeros([1,Nx],dtype=float)
    Longrun_test_1=np.zeros([1,Nx],dtype=float)
    for I in range(0,Nx):
        x_domain[I,0]=xx[0,I]
#        u0[I,0]=data_psi[I,512]
#        u_data[I,0]=data_psi[I,513]
        K[0,I]=kk[0,I]
 #       K2[I,0]=K[I,0]**2.0
        K2[0,I]=kk[0,I]**2.0
        K3[0,I]=kk[0,I]**3.0
        K4[0,I]=kk[0,I]**4.0        
#        kernel_exact[I,0]=T_kernel[0,I]
#        Longrun_test_0[0,I]=data_ps[I,0]
#        Longrun_test_1[0,I]=data_ps[I,n-1]
        
#    x0=np.float32(x0)
#    u0=np.float32(u0)
#    u00=np.zeros([n_dat,Nx], dtype=complex)
    u0r=np.zeros([n_dat,Nx], dtype=float)
    u0i=np.zeros([n_dat,Nx], dtype=float)
    for J in range(0,n_dat):
        for I in range(0,Nx):
            u0r[J,I]=data_psr[I,n_skip*J+n_start]    
            u0i[J,I]=data_psi[I,n_skip*J+n_start]    
    u0r=np.float32(np.real(u0r))
    u0i=np.float32(np.real(u0i))
    
    
#    u0x=(np.roll(u0r,-1,axis=1)-np.roll(u0r,1,axis=1))/(2.0*hx)
#    u0xx=(np.roll(u0r,-1,axis=1)-2.0*u0r+np.roll(u0r,1,axis=1))/(hx*hx)
    
    
#    u0x=np.fft.ifft(-1j*K/1.0*np.fft.fft(u0r))

#    u0xx=np.fft.ifft(-1*K2/1.0*np.fft.fft(u0r))
    
    u0=np.float32(u0)
    u0x=np.float32(u0x)
    u0xx=np.float32(u0xx)
    
    u_datar=np.float32(np.real(u_data))
    u_datai=np.float32(np.imag(u_data))



    N_sp=4
    Nsteps=50000
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

#    model = PDE_NET_Spectral(u0r,u0i,K,K2,K3,K4, u_datar,u_datai, n,N_sp,N_layers,Nx,dt,Losss,The_solutionr,The_solutioni,u0,u0x,u0xx)
    model = PDE_NET_Spectral(u0r,u0i,K,K2,K3,K4, u_datar,u_datai, n,N_sp,N_layers,Nx,dt,Losss,The_solutionr,The_solutioni,u0r,u0x,u0xx,u00,u00x,u00xx)
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    u_pred, sp_weightsr, sp_weightsi, weights, biases= model.predict(u0r,u0i,K,K2,K3,K4,The_solutionr,The_solutioni)
       
    
#Plotting results    
    J=1
    plt.scatter(x_domain,u_pred[J,:], color='black', linewidth=3, label='Prediction')
    plt.plot( x_domain,u_data[J,:], color='red', linewidth=3, label='Exact')
    plt.legend()
    plt.show()

    plt.plot(x_domain, abs(u_data[J,:]-u_pred[J,:]),  color='red', linewidth=3, label='Error')
    plt.legend()
    plt.show()
    
    
#Lets see the fft of the kernel
#    inpt=np.zeros([Nx,1],dtype=float)
#    solution=np.zeros([Nx,1],dtype=float)
#    F_solution=np.zeros([Nx,1],dtype=float) 
#    for I in range(0,Nx):
#        inpt[I,0]=u0[I,0]
#        solution[I,0]=The_solution[I,0]
#        F_solution[I,0]=The_solution[I,n_dat-1]
#    Kernels,F_rhs=The_Kernel(inpt,K,K2, weights, biases,N_layers,dt,Nx)
#    plt.scatter(K[:,0], np.real(Kernels[:,0]),  color='red', linewidth=3, label='FFT of Predicted Kernel')
#    plt.legend()
#    plt.show()#
#    plt.plot(x_domain, abs(Kernels-kernel_exact),  color='red', linewidth=3, label='Error of FFT of Kernel')
#    plt.legend()
#    plt.show()
#    transpsolution=np.zeros([1,Nx],dtype=float)
#    transpkernel=np.zeros([1,Nx],dtype=float)
#    transpF_solution=np.zeros([1,Nx],dtype=float)
#    for I in range(0,Nx):
#        transpsolution[0,I]=solution[I,0]
#        transpkernel[0,I]=Kernels[I,0]
#        transpF_solution[0,I]=F_solution[I,0]
#    for I in range(0,n_dat-1):
#        transpsolution=np.fft.ifft(np.multiply(transpkernel,np.fft.fft(transpsolution)))






plt.scatter(Losss_domain,Losss*10**7, color='black', linewidth=3, label='Loss')
plt.show()

LLosss=np.zeros([Nsteps,1],dtype=float)
for I in range(0,Nsteps):
    LLosss[I,0]=np.log(Losss[I,0])


fig = plt.figure()
plt.scatter(Losss_domain,LLosss, color='black', linewidth=3, label='Loss')
plt.show()
plt.legend()
fig.savefig('kuramoto_fitting_Loss.png')



savemat('Loss.mat',{'L_domain':Losss_domain, 'Loss_y':LLosss})












bb1=biases[0]
Wlast=weights[1]
W0=weights[0]#
nlterm1=Wlast[0,3]*(bb1[0]*W0[1,1])




bb1=biases[0]
bb2=biases[1]
Wlast=weights[2]
W1=weights[1]
W0=weights[0]#
#nlterm=Wlast[0,3]*(W0[0,0]*W0[1,0])+Wlast[0,4]*(W1[0,0]*W1[1,0])
nlterm=Wlast[0,3]*(bb1[0]*W0[1,1])+Wlast[0,4]*(bb2[0]*W1[1,1]+bb2[0]*W1[1,3]*bb1[0]*W0[1,1])#+Wlast[0,4]*(W1[0,3]*bb1[0]*W0[1,0])


bb1=biases[0]
bb2=biases[1]
bb3=biases[2]
Wlast=weights[3]
W2=weights[2]
W1=weights[1]#
W0=weights[0]#

h1=bb1[0]*W0[1,1]
h2=bb2[0]*W1[1,1]+bb2[0]*W1[1,3]*h1
h3=bb3[0]*W2[1,1]+bb3[0]*W2[1,3]*h1+bb3[0]*W2[1,4]*h2

nlterm=Wlast[0,3]*h1+Wlast[0,4]*h2+Wlast[0,5]*h3






bb1=biases[0]
bb2=biases[1]
bb3=biases[2]
bb4=biases[3]
Wlast=weights[4]
W3=weights[3]
W2=weights[2]
W1=weights[1]#
W0=weights[0]#

h1=bb1[0]*W0[1,1]
h2=bb2[0]*W1[1,1]+bb2[0]*W1[1,3]*h1
h3=bb3[0]*W2[1,1]+bb3[0]*W2[1,3]*h1+bb3[0]*W2[1,4]*h2
h4=bb4[0]*W3[1,1]+bb4[0]*W3[1,3]*h1+bb4[0]*W3[1,4]*h2+bb4[0]*W3[1,5]*h3

nlterm=Wlast[0,3]*h1+Wlast[0,4]*h2+Wlast[0,5]*h3+Wlast[0,5]*h4





bb1=biases[0]
bb2=biases[1]
bb3=biases[2]
bb4=biases[3]
Wlast=weights[4]
W3=weights[3]
W2=weights[2]
W1=weights[1]#
W0=weights[0]#

h1=bb1[0]*W0[1,0]
h2=bb2[0]*W1[1,0]+bb2[0]*W1[1,3]*h1
h3=bb3[0]*W2[1,0]+bb3[0]*W2[1,3]*h1+bb3[0]*W2[1,4]*h2
h4=bb4[0]*W3[1,0]+bb4[0]*W3[1,3]*h1+bb4[0]*W3[1,4]*h2+bb4[0]*W3[1,5]*h3

nlterm2=Wlast[0,3]*h1+Wlast[0,4]*h2+Wlast[0,5]*h3+Wlast[0,5]*h4





#sp_weightsr= real parts of weights of Linear operator
#sp_weightsi= imaginary part of weights of Linear operator
#u\partial_x t-squared term = nlterm
#u-squared term = nlterm2

wr=sp_weightsr[0]
wi=sp_weightsi[0]

print('The learned PDE is $\partial_t u = $:')
print('R.H.S. of Real part:')
print(wr[0,0],'u_x +',wr[0,1],'u_xx+',wr[0,2],'u_xxx',wr[0,3],'u_xxxx')
print('R.H.S. of Imaginary part:')
print(wi[0,0],'u_x +',wi[0,1],'u_xx+',wi[0,2],'u_xxx',wi[0,3],'u_xxxx')
print('non-linear term')
print(nlterm,'u u_x +...')
print('quadratic nonlinear term')
print(nlterm2,'u^2+...') #This is more pertinent to the Nonlinear schrodinger equation


"""
thelossdata0 = scipy.io.loadmat('Loss0.mat')
thelossdata1 = scipy.io.loadmat('Loss1.mat')
thelossdata2 = scipy.io.loadmat('Loss2.mat')
xxlos0 = thelossdata0['L_domain']
datalos0 = thelossdata0['Loss_y']
xxlos1 = thelossdata1['L_domain']
datalos1 = thelossdata1['Loss_y']
xxlos2 = thelossdata2['L_domain']
datalos2 = thelossdata2['Loss_y']

fig = plt.figure()
plt.plot(xxlos0,datalos0, color='black', linewidth=3, label="Fwrd-Euler")
plt.legend() 
plt.plot(xxlos0,datalos1, color='blue', linewidth=3, label='SP1')
plt.legend() 
plt.plot(xxlos0,datalos2, color='red', linewidth=3, label='SP2')
plt.legend()
#fig.savefig('Loss_compilation_heat_50_dt.png')
#plt.show()















thelossdata0 = scipy.io.loadmat('Loss0.mat')
thelossdata1 = scipy.io.loadmat('Loss1.mat')
thelossdata2 = scipy.io.loadmat('Loss2.mat')
xxlos0 = thelossdata0['L_domain']
datalos0 = thelossdata0['Loss_y']
xxlos1 = thelossdata1['L_domain']
datalos1 = thelossdata1['Loss_y']
xxlos2 = thelossdata2['L_domain']
datalos2 = thelossdata2['Loss_y']


newdomain=np.zeros([100,1],dtype=float)
newdomain2=np.zeros([50,1],dtype=float)
newloss0=np.zeros([100,1],dtype=float)
newloss1=np.zeros([100,1],dtype=float)
newloss2=np.zeros([50,1],dtype=float)
for I in range(0,100):
    newdomain[I,0]=xxlos0[600*I,0]
    newloss0[I,0]=datalos0[600*I,0]
    newloss1[I,0]=datalos1[600*I,0]
    
for I in range(0,50):
    newdomain2[I,0]=xxlos0[1200*I,0]
    newloss2[I,0]=datalos2[1200*I,0]

fig = plt.figure()
plt.plot(newdomain,newloss0, color='black', linewidth=3, label="Fwrd-Euler")
plt.legend() 
plt.plot(newdomain,newloss1, color='blue', linewidth=3, label='SP1')
plt.legend() 
plt.scatter(newdomain2,newloss2, color='red', linewidth=2, label='SP2')
plt.legend()
fig.savefig('Loss_compilation_heat.png')












thelossdata0 = scipy.io.loadmat('Loss0.mat')
thelossdata1 = scipy.io.loadmat('Loss1.mat')
thelossdata2 = scipy.io.loadmat('Loss2.mat')
xxlos0 = thelossdata0['L_domain']
datalos0 = thelossdata0['Loss_y']
xxlos1 = thelossdata1['L_domain']
datalos1 = thelossdata1['Loss_y']
xxlos2 = thelossdata2['L_domain']
datalos2 = thelossdata2['Loss_y']

fig = plt.figure()
plt.plot(xxlos0,datalos0, color='black', linewidth=3, label="Fwrd-Euler")
plt.legend() 
plt.plot(xxlos0,datalos1, color='blue', linewidth=3, label='CN')
plt.legend() 
plt.plot(xxlos0,datalos2, color='red', linewidth=3, label='Spectral')
plt.legend()
fig.savefig('Loss_compilation_heat_200_dt.png')












thelossdata0 = scipy.io.loadmat('Loss0.mat')
thelossdata1 = scipy.io.loadmat('Loss4.mat')
thelossdata2 = scipy.io.loadmat('Loss1.mat')
thelossdata3 = scipy.io.loadmat('Loss5.mat')
xxlos0 = thelossdata0['L_domain']
datalos0 = thelossdata0['Loss_y']
xxlos1 = thelossdata1['L_domain']
datalos1 = thelossdata1['Loss_y']
xxlos2 = thelossdata2['L_domain']
datalos2 = thelossdata2['Loss_y']
xxlos3 = thelossdata3['L_domain']
datalos3 = thelossdata3['Loss_y']


fig = plt.figure()
plt.plot(xxlos0,datalos0, color='black', linewidth=3, label="Fwrd-Euler")
plt.legend() 
plt.plot(xxlos0,datalos1, color='blue', linewidth=3, label='CN')
plt.legend() 
plt.plot(xxlos0,datalos2, color='red', linewidth=3, label='Spectral')
plt.legend()
plt.plot(xxlos0,datalos3, color='red', linewidth=3, label='Spectral')
plt.legend()
fig.savefig('Loss_compilation_heat2.png')














#Wlast=weights[1]
#W0=weights[0]
#nlterm=Wlast[0,3]*(W0[0,0]*W0[1,0])




"""


