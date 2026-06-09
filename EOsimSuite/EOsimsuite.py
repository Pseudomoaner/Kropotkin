# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""    
Define equations that will get input into solve_ivp. Each represents
placing the specified EO system into a different experimental setting.
"""
#Macarthur model (with biotic resources)

def macarthur(t, y, EOparams, sysParams, noSpec, noRes, impactFunc, sensFunc):
    Ks = sysParams['Ks']
    vs = sysParams['vs']
    ts = sysParams['ts']
    
    rs = y[:noRes]
    specs = y[noRes:]

    dr = np.zeros(rs.shape)
    for bet in range(noSpec):
        dr += impactFunc(rs,bet,EOparams)*specs[bet]
    for rho in range(noRes):
        dr[rho] += (Ks[rho] - rs[rho])*vs[rho]*rs[rho]/Ks[rho] #Sigma for a logistic resource  
    ds = np.array([(sensFunc(rs,alph,EOparams) - ts[alph]) * specs[alph] for alph in range(noSpec)])

    dY = np.concatenate((dr,ds),axis=0)
    return dY

#Chemostat model
def chemostat(t, y, EOparams, sysParams, noSpec, noRes, impactFunc, sensFunc):
    rs = y[:noRes]
    specs = y[noRes:]

    dr = np.zeros(rs.shape)
    for bet in range(noSpec):
        dr += impactFunc(rs,bet,EOparams)*specs[bet]
    dr += (sysParams['rIn'] - rs)*sysParams['D'] #Sigma for a chemostat   
    ds = np.array([(sensFunc(rs,alph,EOparams) - sysParams['D']) * specs[alph] for alph in range(noSpec)])

    dY = np.concatenate((dr,ds),axis=0)
    return dY

#Batch culture model
def batchReact(t, y, EOparams, sysParams, noSpec, noRes, impactFunc, sensFunc):    
    rs = y[:noRes]
    specs = y[noRes:]

    dr = np.zeros(rs.shape)
    for bet in range(noSpec):
        dr += impactFunc(rs,bet,EOparams)*specs[bet]

    ds = np.array([sensFunc(rs,alph,EOparams) * specs[alph] for alph in range(noSpec)])
    
    dY = np.concatenate((dr,ds),axis=0)
    return dY

#Colony model
def colony(t, y, EOparams, sysParams, noSpec, noRes, noX, impactFunc, sensFunc):
    rs = y[:noRes*noX]
    specs = y[noRes*noX:]
    
    rTile = np.reshape(rs,(noX,noRes))
    specTile = np.reshape(specs,(noX,noSpec))
    
    #Changes in frequency of different pops. based on current fitnesses
    dsFree = np.array([[sensFunc(rTile[chi,:],alph,EOparams) * specTile[chi,alph] for alph in range(noSpec)] for chi in range(noX)]) #Fitnesses under unconfined conditions
    meanFits = np.expand_dims(np.sum(np.multiply(specTile,dsFree),axis=1),axis=1)
    meanFitsTile = np.broadcast_to(meanFits,(noX,noSpec))
    selects = np.multiply(np.divide(dsFree-meanFitsTile,meanFitsTile),specTile)   
    
    #Also need to account for diffusion over space
    specDiffs = diffModule(specTile, sysParams['specDiffConst'], sysParams['dX'], 'periodic', 'periodic')
    
    ds = selects.flatten() + specDiffs.flatten()
    
    #Resources - autogenic mechanisms
    autogenics = np.zeros(rTile.shape)
    for chi in range(noX):
        for bet in range(noSpec):
            autogenics[chi,:] += impactFunc(rTile[chi,:],bet,EOparams)*specTile[chi,bet]
    
    #And diffusion
    resDiffs = diffModule(rTile, sysParams['resDiffConst'], sysParams['dX'], 'periodic', 'periodic')
    
    #And chemostat-like term from external resources
    sigs = (np.broadcast_to(sysParams['rIn'],(noX,noRes)) - rTile)*sysParams['D'] #Sigma for a chemostat
    
    dr = autogenics.flatten() + resDiffs.flatten() + sigs.flatten()
    
    dY = np.concatenate((dr,ds),axis=0)
    return dY

"""
1-dimensional reaction advection/diffusion equation stuff. Bounds indicate the 
boundary conditions for the pde: 
    -'fixed' sets them to remain constant 
    -'noflux' reflects material back into the domain
    -'free' allows material to leak out of the system (assuming the 
    concentrations outside the boudary closely resemble those at the boundary)
    -'periodic' allows wrap-around of material to the other boundary.
"""

#Define the module that calculates the diffusive component of the convection-diffusion equation
def diffModule(c,D,dX,bound1,bound2):
    #Simplest case - used for colony models. Periodic BCs.
    if bound1 == 'periodic' and bound2 == 'periodic':
        cRight = np.roll(c,1,axis=0)
        cLeft = np.roll(c,-1,axis=0)
        dC = D*(cRight + cLeft - 2*c)/(dX**2)
    else:
        print('Implement this code!')
        dC = np.zeros(c.shape())
        """
        dC = np.zeros(np.size(c))
        for i in range(len(c)):
            if i == 0:
                if bound1 == 'fixed':
                    dC[i] = 0
                    elif bound1 == 'free':
                        dC[i] = D*(-2*c[1] + c[0] + c[2])/(dX**2)
                    elif bound1 == 'noflux':
                        dC[i] = D*(-c[0] + c[1])/(dX**2)
                    elif bound1 == 'periodic':
                dC[i] = D*(-2*c[0] + c[1] + c[-1])/(dX**2)
            elif i == len(c) - 1:
                if bound2 == 'fixed':
                    dC[i] = 0
                elif bound2 == 'free':
                    dC[i] = D*(-2*c[-2] + c[-1] + c[-3])/(dX**2)
                elif bound2 == 'noflux':
                    dC[i] = D*(-c[-1] + c[-2])/(dX**2)
                elif bound2 == 'periodic':
                    dC[i] = D*(-2*c[-1] + c[0] + c[-2])/(dX**2)
            else:
                dC[i] = D*(-2*c[i] + c[i+1] + c[i-1])/(dX**2)
            """
    
    return dC

#Define the module that calculates the convective component of the convection-diffusion equation
def convectModule(c,V,dX,bound1,bound2):
    dC = np.zeros(np.size(c))
    for i in range(len(c)):
        if i == 0:
            if bound1 == 'fixed':
                dC[i] = 0
            elif bound1 == 'free':
                dC[i] = -V*(c[i+1] - c[i])/dX 
            elif bound1 == 'noflux':
                dC[i] = -V*(c[i] + c[i+1])/2
            elif bound1 == 'periodic':
                dC[i] = -V*(c[-1] - c[i+1])/(2*dX)
        elif i == len(c) - 1:
            if bound2 == 'fixed':
                dC[i] = 0
            elif bound2 == 'free':
                dC[i] = V*(c[i-1]-c[i])/dX 
            elif bound2 == 'noflux':
                dC[i] = V*(c[i] + c[i-1])/2
            elif bound2 == 'periodic':
                dC[i] = V*(c[i-1] - c[0])/(2*dX)
        else:
            dC[i] = V*(c[i-1] - c[i+1])/(2*dX)
            
    return dC

class EOsimsuite:
    """
    Initialisation function sets the minimum inputs necessary for a numerical
    simulation. So sensitivity function gradient, instantaneous interactions 
    etc. don't need to be set here.
    """
    def __init__(self,impactFunc,sensFunc,EOparams,sysParams,B0,R0):
        self.impactFunc = impactFunc
        self.sensFunc = sensFunc
        self.EOparams = EOparams
        self.sysParams = sysParams
        self.set_initConds(B0, R0, False)
        self.set_timeSettings() #Use default time settings - user can overwrite later
        
    """
    Simulation settings setup (initial conditions etc.)
    """
    def set_initConds(self,B0,R0,spatial):
        if spatial: #If a 1D spatial model (flowcell or colony)
            Bnorm = B0/sum(B0) #Need to ensure populations add to 1 for the colony model; perhaps this is not such a good idea for the flowcell but keep for now
            Bnoise = np.broadcast_to(Bnorm,(self.noX,self.noS)) + np.random.normal(0,self.sysParams['specNoise'],(self.noX,self.noS))
            expandB = np.expand_dims(np.sum(Bnoise,axis=1),axis=1)
            
            self.B0 = np.divide(Bnoise,np.broadcast_to(expandB,(self.noX,self.noS))).flatten()
            
            Rnoise = np.broadcast_to(R0,(self.noX,self.noR)) + np.random.normal(0,self.sysParams['resNoise'],(self.noX,self.noR))
            
            self.R0 = Rnoise.flatten()
        else: #If a well-mixed model (chemostat, batch or serial transfer)
            self.B0 = B0
            self.R0 = R0
        
        self.noS = len(B0)
        self.noR = len(R0)
    
    #Sets time dimensions
    def set_timeSettings(self,**kwargs):
        if kwargs.get('tSpan'):
            self.tSpan = kwargs.get('tSpan')
        else:
            self.tSpan = [0,100]
        
        if kwargs.get('dt'):
            self.dt = kwargs.get('dt')
        else:
            self.dt = 0.1
        
        self.noT = int((self.tSpan[1] - self.tSpan[0])/self.dt)
    
    #Sets spatial dimensions (only used for colony and flowcell models)
    def set_spaceSettings(self,**kwargs):
        if kwargs.get('L'):
            self.L = kwargs.get('L')
        else:
            self.L = 50
        
        if kwargs.get('dx'):
            self.dx = kwargs.get('dx')
        else:
            self.dx = 0.5
        
        self.noX = int(self.L/self.dx)
    
    #Sets dilution settings (only used for serial transfer and chemostat models)
    def set_dilSettings(self,**kwargs):
        self.sysParams['D'] = kwargs['D'] #D must be available for any diluted system, represents the long-term dilution rate
        
        if kwargs.get('transT'): #Dilution frequency for serial transfers
            if kwargs['transT'] * kwargs['D'] > 1:
                raise AssertionError('Dilution factor is greater than 1! Reduce D or delT.')
            else:
                self.sysParams['transT'] = kwargs['transT']
                self.sysParams['dilFac'] = kwargs['transT'] * kwargs['D']
                totT = self.tSpan[1]-self.tSpan[0]
                self.sysParams['noTrans'] = int(np.ceil(totT/kwargs['transT']))
    
    #Sets the starting resource concentrations to give rise to a stable point (if defined and feasible)
    def set_stationaryStart(self,method):
        if not hasattr(self,'rStar'):
            raise AttributeError('rStar not available! Try running set_findRstar() first.')
        if  not hasattr(self,'xStar'):
            self.findXstar(method)
            
        if sum(self.rStar > 0) < len(self.rStar):
            raise AssertionError('Stationary resource distribution is not feasible!')
        if sum(self.xStar > 0) < len(self.xStar):
            raise AssertionError('Stationary species distribution is not feasible!')
        
        self.R0 = self.rStar
        self.B0 = self.xStar + np.random.normal(0,0.0001,self.B0.shape)
                
    """
    Derived quantities setup (gradient of sensitivity function, instantaneous 
    interactions etc.)
    """
    def set_instInteract(self,instInteract):
        self.instInteract = instInteract
    
    def set_gradSens(self,gradSens):
        self.gradSens = gradSens
        
    def set_intrinsicGR(self,intrinsicGR):
        self.intrinsicGR = intrinsicGR
        
    def set_findRstar(self,findRstar):
        self.rStarFunc = findRstar
        self.rStar = findRstar(self.EOparams,self.sysParams)
    
    def findXstar(self,method):
        if not hasattr(self,'rStar'):
            raise AttributeError('rStar not available! Try running set_findRstar() first.')
        
        #Methods are: 'Exact' (based on impact functions), 'eEO' (solution to eEO equation)
        if method == 'Exact':
            F = np.array([self.impactFunc(self.rStar,alphInd,self.EOparams) for alphInd in range(self.noS)])
            sig = self.sysParams['D'] * (self.sysParams['rIn']-self.rStar)
        
            self.xStar = -np.matmul(np.linalg.inv(F.T),sig)
        elif method == 'eEO':
            intAr = self.assembleIntMat(self.rStar)
            GRvec = self.assembleGRvec(self.rStar)
        
            self.xStar = -np.matmul(np.linalg.inv(intAr),GRvec)
        
    #Assembles and returns an equivalent interaction matrix for a system at equilibrium at rStar
    def assembleIntMat(self,rStar):
        if not hasattr(self,'instInteract'):
            raise AttributeError('instInteract not available! Try running set_instInteract() first.')
        
        return np.array([[self.instInteract(rStar,alphInd,betInd,self.EOparams) 
                         for betInd in range(self.noS)] 
                         for alphInd in range(self.noS)])
    
    #Assembles a vector of intrinsic growth rates
    def assembleGRvec(self,rStar):
        if not hasattr(self,'intrinsicGR'):
            raise AttributeError('intrinsicGR not available! Try running set_intrinsicGR() first.')
        
        return np.array([self.intrinsicGR(rStar,alphInd,self.EOparams,self.sysParams) 
                         for alphInd in range(self.noS)])
    
    """
    Function that actually runs the specified numerical integration
    """
    def simEOmodel(self,modelType):
        if modelType == 'Chemostat' or modelType == 'Batch' or modelType == 'SerialTransfer' or modelType == 'MacArthur':
            inArgs = (self.EOparams,self.sysParams,self.noS,self.noR,self.impactFunc,self.sensFunc)
        elif modelType == 'Flowcell' or modelType == 'Colony':
            inArgs = (self.EOparams,self.sysParams,self.noS,self.noR,self.noX,self.impactFunc,self.sensFunc)
        
        if modelType == 'Chemostat':
            tSteps = np.arange(self.tSpan[0],self.tSpan[1],self.dt)
            self.solution = solve_ivp(fun=chemostat, t_span=self.tSpan, y0=np.concatenate((self.R0,self.B0),axis=0), t_eval=tSteps, args=inArgs)
        
            self.resTimeseries = self.solution.y[:self.noR,:]
            self.specTimeseries = self.solution.y[self.noR:,:]
        
        elif modelType == 'MacArthur':
            tSteps = np.arange(self.tSpan[0],self.tSpan[1],self.dt)
            self.solution = solve_ivp(fun=macarthur, t_span=self.tSpan, y0=np.concatenate((self.R0,self.B0),axis=0), t_eval=tSteps, args=inArgs)
        
            self.resTimeseries = self.solution.y[:self.noR,:]
            self.specTimeseries = self.solution.y[self.noR:,:]
            
        elif modelType == 'Batch':
            tSteps = np.arange(self.tSpan[0],self.tSpan[1],self.dt)
            self.solution = solve_ivp(fun=batchReact, t_span=self.tSpan, y0=np.concatenate((self.R0,self.B0),axis=0), t_eval=tSteps, args=inArgs)
        
            self.resTimeseries = self.solution.y[:self.noR,:]
            self.specTimeseries = self.solution.y[self.noR:,:]
            
        elif modelType == 'SerialTransfer':
            
            #Run batch culture model multiple times, diluting at end of each cycle
            transSpan = [0,self.sysParams['transT']]
            tSteps = np.arange(transSpan[0],transSpan[1],self.dt)
            initConds = np.concatenate((self.R0,self.B0),axis=0)
            
            self.solution = []
            
            for transInd in range(self.sysParams['noTrans']):
                self.solution.append(solve_ivp(fun=batchReact, t_span=transSpan, y0=initConds, t_eval=tSteps, args=inArgs))
                
                #Run dilution for next set of initial conditions
                dilRes = self.solution[transInd].y[:len(self.R0),-1]*(1-self.sysParams['dilFac']) + self.sysParams['dilFac']*self.sysParams['rIn']
                dilSpec = self.solution[transInd].y[len(self.R0):,-1]*(1-self.sysParams['dilFac'])
                initConds = np.concatenate((dilRes,dilSpec),axis=0)
            
            #Collate timeseries from collection of solutions
            self.resTimeseries = np.concatenate([self.solution[transInd].y[:self.noR,:] for transInd in range(self.sysParams['noTrans'])],axis=1)
            self.specTimeseries = np.concatenate([self.solution[transInd].y[self.noR:,:] for transInd in range(self.sysParams['noTrans'])],axis=1)
        
        elif modelType == 'Colony':
            
            #Simulate an expanding colony model, based on Muller et al. PNAS (2014)
            tSteps = np.arange(self.tSpan[0],self.tSpan[1],self.dt)
            self.solution = solve_ivp(fun=colony, t_span=self.tSpan, y0=np.concatenate((self.R0,self.B0),axis=0), t_eval=tSteps, args=inArgs)
        
            resDat = self.solution.y[:self.noR*self.noX,:]
            specDat = self.solution.y[self.noR*self.noX:,:]
            
            self.resTimeseries = np.transpose(np.reshape(resDat, (self.noX,self.noR,self.noT)),(1,2,0))
            self.specTimeseries = np.transpose(np.reshape(specDat, (self.noX,self.noS,self.noT)),(1,2,0))
            
        self.modelType = modelType #Store the type of model that was requested
        
    """
    Visualisation functions
    """
    def genInteractSpace(self,maxRho,maxThet,alph,bet):
        sampPtsRho = 200
        sampPtsThet = 200

        return np.array([[self.instInteract(np.array([rho,thet]),alph,bet,self.EOparams) 
                          for thet in np.linspace(0,maxThet,sampPtsThet)] 
                          for rho in np.linspace(0,maxRho,sampPtsRho)])
    
    def plotAllInteractSpaces(self,maxRho,maxThet,plotTraj):
        minA = float('inf')
        maxA = float('-inf')
        
        fig1, axAr = plt.subplots(nrows = self.noS, ncols = self.noS, constrained_layout = True)
        
        for alph in range(len(self.B0)):
            for bet in range(len(self.B0)):
                currSpace = self.genInteractSpace(maxRho, maxThet, alph, bet)
                
                if currSpace.max() > maxA:
                    maxA = currSpace.max()
                if currSpace.min() < minA:
                    minA = currSpace.min()
        
        cLim = max(abs(minA),abs(maxA))
        
        for alph in range(len(self.B0)):
            for bet in range(len(self.B0)):
                currSpace = self.genInteractSpace(maxRho, maxThet, alph, bet)
                #axAr[alph,bet].imshow(currSpace.T,cmap='BrBG',aspect='auto',origin='lower',extent=[0,maxRho,0,maxThet],vmax=cLim,vmin=-cLim)
                axAr[alph,bet].contour(currSpace.T,31,cmap='BrBG',origin='lower',extent=[0,maxRho,0,maxThet],vmax=cLim,vmin=-cLim)
                if plotTraj:
                    axAr[alph,bet].plot(self.resTimeseries[0,:],self.resTimeseries[1,:],'k')
            
    
    def plotSolutionTimecourse(self):
        if self.modelType == 'Chemostat' or self.modelType == 'Batch' or self.modelType == 'SerialTransfer' or self.modelType == 'MacArthur':
            tSteps = np.arange(self.tSpan[0],self.tSpan[1],self.dt)
        
            plt.plot(tSteps,self.resTimeseries.T,'r')
            plt.plot(tSteps,self.specTimeseries.T,'b')
            #plt.plot(tSteps,np.sum(self.resTimeseries,axis=0)+np.sum(self.specTimeseries,axis=0),'k')
        
            plt.xlabel('Time')
            plt.ylabel('Abundance')
        elif self.modelType == 'Colony' or self.modelType == 'Flowcell':
            fig1, axAr = plt.subplots(nrows = max((self.noS,self.noR)), ncols = 2, constrained_layout = True)
            
            if self.noS == 1:
                axAr[0].imshow(self.specTimeseries[0,:,:],cmap='Blues',aspect='auto', origin='lower',extent=[0,self.L,self.tSpan[0],self.tSpan[1]],vmin=0,vmax=1)
            else:
                for alph in range(self.noS):
                    axAr[alph,0].imshow(self.specTimeseries[alph,:,:],cmap='Blues',aspect='auto', origin='lower',extent=[0,self.L,self.tSpan[0],self.tSpan[1]],vmin=0,vmax=1)
            
            if self.noR == 1:
                axAr[1].imshow(self.resTimeseries[0,:,:],cmap='Reds',aspect='auto', origin='lower',extent=[0,self.L,self.tSpan[0],self.tSpan[1]])
            else:
                for rho in range(self.noR):
                    axAr[rho,1].imshow(self.resTimeseries[rho,:,:],cmap='Reds',aspect='auto', origin='lower',extent=[0,self.L,self.tSpan[0],self.tSpan[1]])