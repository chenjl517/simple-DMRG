import numpy as np
from scipy.sparse.linalg import eigsh
from globalConfig import basicCell,Sx,Sy,Sz
from utils import transformBase

HEnv=np.copy(basicCell) # environment
HSys=np.copy(basicCell) # system

looptime=1
truncatureNum=3

Hsuper=np.kron(basicCell,np.identity(4))+\
    np.kron(np.identity(2),np.kron(basicCell,np.identity(2)))+\
    np.kron(np.identity(4),basicCell)

N=4
lastE=0
for i in range(looptime):

    v0=np.random.random(np.shape(Hsuper)[0]);v0/=np.sqrt(np.linalg.norm(v0)) #gen real initial vector for lanczos to start
    w,phi0=eigsh(Hsuper,k=1,v0=v0) # get the smallest eigenvalue and eigenvector by lanczos
    E_0=(w[0]-lastE)/2
    lastE=w[0]
    print("E0_progress=",E_0)
    phi0=np.reshape(phi0,(int(np.sqrt(phi0.shape[0])),-1))
    # construct density matrix

    ## construct density matrix for the system
    rhoSys=np.tensordot(phi0,np.conj(phi0),(1,1))
    # rhoSys=phi0@np.conj(np.transpose(phi0))
    w,USys=np.linalg.eigh(-rhoSys) # - rhoSys,to get eigenvalue in the inverse order.
    USys=USys[:,0:truncatureNum]

    ## construct density matrix for the environment
    rhoEnv=np.tensordot(np.conj(phi0),phi0,(0,0))
    # rhoEnv=np.conj(np.transpose(phi0))@phi0
    print("最大虚部：",np.max(np.imag(rhoEnv)))
    w,UEnv=np.linalg.eigh(-rhoEnv)
    UEnv=UEnv[:,0:truncatureNum]

    # construct related operator in the new base
    nowI=np.identity(int(np.shape(phi0)[0]/2))

    ## System
    SxNow=np.kron(nowI,Sx)
    SyNow=np.kron(nowI,Sy)
    SzNow=np.kron(nowI,Sz)
    SInnerEdgeSnew=np.kron(transformBase(USys,SxNow),Sx)+\
                   np.kron(transformBase(USys,SyNow),Sy)+\
                   np.kron(transformBase(USys,SzNow),Sz)
    HSys=np.kron(transformBase(USys,HSys),np.identity(2))+SInnerEdgeSnew

    ## Env
    SxNow = np.kron(Sx,nowI)
    SyNow = np.kron(Sy,nowI)
    SzNow = np.kron(Sz,nowI)
    SInnerEdgeSnew=np.kron(Sx,transformBase(UEnv,SxNow))+\
                   np.kron(Sy,transformBase(UEnv,SyNow))+\
                   np.kron(Sz,transformBase(UEnv,SzNow))
    HEnv=np.kron(np.identity(2),transformBase(UEnv,HEnv))+SInnerEdgeSnew

    ##super
    baseNum=np.shape(HEnv)[0]
    Hsuper=np.kron(HSys,np.identity(baseNum))+\
        np.kron(np.identity(int(baseNum/2)),np.kron(basicCell,np.identity(int(baseNum/2))))+\
        np.kron(np.identity(baseNum),HEnv)
    print(Hsuper)
    print("Hsuper最大虚部：",np.max(np.imag(Hsuper)))


# calculate ground-state energy
(finalNE0,)=eigsh(Hsuper,k=1,return_eigenvectors=False)
print("E0=",(finalNE0-lastE)/2)